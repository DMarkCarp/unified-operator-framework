# Git Cheat Sheet — Research Workflow

A quick reference for managing research projects with git and GitHub.
Built for Windows PowerShell. The commands are identical on Mac/Linux.

---

## Mental model

```
Working Directory  →  Staging Area  →  Local Repo  →  GitHub (remote)
   (files)           (git add)       (git commit)     (git push)
```

Commits are snapshots. Branches are parallel timelines. Tags are bookmarks.
GitHub is just a copy of your local repo that lives on the internet.

---

## Everyday commands (use these constantly)

```
git status                    # What's going on? Always safe to run.
git add .                     # Stage all changes for the next commit
git commit -m "message"       # Snapshot the staged changes
git push                      # Send commits to GitHub
git pull                      # Fetch GitHub's commits into local
git log --oneline             # Show recent commit history
```

When in doubt, run `git status`. It tells you the current state and usually
suggests what to do next.

---

## Setting up a new project (one-time per project)

```
cd C:\path\to\project
git init                      # Create the repository
# (create .gitignore — see below)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git add .
git commit -m "Initial commit"
git tag -a v1.0-baseline -m "Starting point"
```

Then create a private repo on github.com (do NOT initialize with README) and:

```
git remote add origin https://github.com/USERNAME/REPONAME.git
git branch -M main
git push -u origin main
git push --tags
```

---

## Cloning to a new machine

```
cd C:\desired\parent\directory
git clone https://github.com/USERNAME/REPONAME.git
cd REPONAME
# (recreate Python venv from requirements.txt — see Python section)
```

---

## Branches (use for any non-trivial change)

```
git checkout -b feature-name  # Create new branch and switch to it
git branch                    # List branches; * shows current
git checkout main             # Switch back to main
git merge feature-name        # Merge feature-name into current branch
git branch -d feature-name    # Delete the branch after merging
```

Workflow: create a branch, make changes, commit, test, merge to main when ready.
This keeps main always in a known-good state.

---

## Tags (mark important reference points)

```
git tag -a v1.0-name -m "Description"   # Create annotated tag
git tag                                  # List all tags
git push --tags                          # Send tags to GitHub
git checkout v1.0-name                   # Return to a tagged state (read-only)
```

Use tags before risky changes so you can always return to a known state.

---

## Undoing things

```
git reset                         # Unstage everything (files unchanged)
git restore <file>                # Discard changes to a file (BE CAREFUL)
git commit --amend -m "new msg"   # Fix the last commit's message
git revert <commit-hash>          # Undo a commit by adding a new commit
```

`git reset` is safe (only affects staging). `git restore` deletes your changes.
`git revert` is the safe way to undo a commit because it preserves history.

---

## Inspecting

```
git status                  # Current state
git log --oneline           # Compact commit history
git log -p                  # Detailed history with changes
git diff                    # What's changed since last commit
git diff --staged           # What's staged for the next commit
git show <hash>             # Details of a specific commit
git remote -v               # Show configured remotes
```

---

## .gitignore essentials

A plain text file in the project root listing patterns to skip.
Create it with PowerShell using `New-Item` + `Add-Content` (NOT `echo` —
PowerShell's echo creates a wrong encoding that git can't read).

Typical content for a Python research project:

```
.venv/
__pycache__/
.claude/
.ipynb_checkpoints/
*.pyc
*.zip
.DS_Store
```

---

## Python environment management (related but not git)

Virtual environments are per-machine. They don't transfer via git.
You transfer `requirements.txt` and recreate the venv on each machine.

```
python -m venv .venv                              # Create venv
.venv\Scripts\Activate.ps1                        # Activate (Windows)
pip install -r requirements.txt                   # Install packages
pip freeze > requirements.txt                     # Save current versions
python -m ipykernel install --user --name NAME    # Register Jupyter kernel
```

When transferring a project to a new machine:
1. Clone the repo
2. Create a fresh venv
3. Install from requirements.txt
4. Register the Jupyter kernel
5. Verify your code runs and produces the same results

---

## My typical workflow for a research project

1. Start fresh:
   ```
   mkdir project-name
   cd project-name
   git init
   ```

2. Add the basics:
   ```
   # write .gitignore
   # write a brief README.md describing the project
   git add .
   git commit -m "Initial project setup"
   ```

3. Push to GitHub:
   ```
   # create private repo on github.com
   git remote add origin https://github.com/USERNAME/project-name.git
   git branch -M main
   git push -u origin main
   ```

4. Daily work:
   ```
   git status               # Check before starting
   # do work
   git add .
   git commit -m "Did X"
   git push                 # End of session
   ```

5. Before risky changes:
   ```
   git tag -a v0.5-checkpoint -m "Before adding feature Y"
   git push --tags
   git checkout -b feature-y
   # do experimental work on the branch
   ```

6. When it works:
   ```
   git checkout main
   git merge feature-y
   git push
   ```

---

## When things go wrong

- **Confused about state:** `git status` first, always.
- **Want to undo recent local work:** `git reset --hard HEAD` (DESTRUCTIVE).
- **Want to undo a commit safely:** `git revert <hash>`.
- **Accidentally staged something:** `git reset` (without arguments).
- **Want to return to a tagged state:** `git checkout v1.0-name`, then
  `git checkout -b new-branch` to start working from there.
- **GitHub rejects push:** usually means someone else pushed first.
  Run `git pull --rebase` and try again.
- **Weird encoding error:** PowerShell `echo` problem. Use `New-Item` and
  `Add-Content` instead.
- **Authentication fails:** create a Personal Access Token in GitHub settings
  (Developer settings → Tokens). Paste the token when prompted for password.

---

## What experienced people actually remember vs. look up

**Memorized:** `status`, `add`, `commit`, `push`, `pull`, `log`, basic branch
operations.

**Looked up regularly:** anything destructive (reset, revert, force-push),
weird state recovery, less-common commands.

**Asked AI:** anything you only do once every few months. There's no shame
in this — it's the modern workflow.

---

Print this. Tape it next to your monitor. Refer to it for six months.
Eventually you'll stop needing the everyday section, then the project-setup
section, then most of it. The "When things go wrong" section is the one
you'll keep coming back to.
