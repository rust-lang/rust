# Using Git

<!-- toc -->

The Rust project uses [Git] to manage its source code. In order to
contribute, you'll need some familiarity with its features so that your changes
can be incorporated into the compiler.

[Git]: https://git-scm.com

The goal of this page is to cover some of the more common questions and
problems new contributors face. Although some Git basics will be covered here,
if you  find that this is still a little too fast for you, it might make sense
to first read some introductions to Git, such as the Beginner and Getting
started sections of [this tutorial from Atlassian][atlassian-git]. GitHub also
provides [documentation] and [guides] for beginners, or you can consult the
more in depth [book from Git].

[book from Git]: https://git-scm.com/book/en/v2/
[atlassian-git]: https://www.atlassian.com/git/tutorials/what-is-version-control
[documentation]: https://docs.github.com/en/github/getting-started-with-github/set-up-git
[guides]: https://guides.github.com/introduction/git-handbook/

## Prerequisites

We'll assume that you've installed Git, forked [rust-lang/rust], and cloned the
forked repo to your PC. We'll use the command line interface to interact
with Git; there are also a number of GUIs and IDE integrations that can
generally do the same things.

[rust-lang/rust]: https://github.com/rust-lang/rust

If you've cloned your fork, then you will be able to reference it with `origin`
in your local repo. It may be helpful to also set up a remote for the official
rust-lang/rust repo via

```sh
git remote add upstream https://github.com/rust-lang/rust.git
```

if you're using HTTPS, or

```sh
git remote add upstream git@github.com:rust-lang/rust.git
```

if you're using SSH.

## Standard Process

Below is the normal procedure that you're likely to use for most minor changes
and PRs:

 1. Ensure that you're making your changes on top of master:
 `git checkout master`.
 2. Get the latest changes from the Rust repo: `git pull upstream master`.
 3. Make a new branch for your change: `git checkout -b issue-12345-fix`.
 4. Make some changes to the repo and test them.
 5. Stage your changes via `git add src/changed/file.rs src/another/change.rs`
 and then commit them with `git commit`. Of course, making intermediate commits
 may be a good idea as well. Avoid `git add .`, as it makes it too easy to
 unintentionally commit changes that should not be committed, such as submodule
 updates. You can use `git status` to check if there are any files you forgot
 to stage.
 6. Push your changes to your fork: `git push --set-upstream origin issue-12345-fix`.
 7. [Open a PR][ghpullrequest] from your fork to rust-lang/rust's master branch.

[ghpullrequest]: https://guides.github.com/activities/forking/#making-a-pull-request

If your reviewer requests changes, the procedure for those changes looks much
the same, with some steps skipped:

 1. Ensure that you're making changes to the most recent version of your code:
 `git checkout issue-12345-fix`.
 2. Make, stage, and commit your additional changes just like before.
 3. Push those changes to your fork: `git push`.

## Troubleshooting git issues

You don't need to clone `rust-lang/rust` from scratch if it's out of date!
Even if you think you've messed it up beyond repair, there are ways to fix
the git state that don't require downloading the whole repository again.
Here are some common issues you might run into:

### I deleted my fork on GitHub!

This is not a problem from git's perspective. If you run `git remote -v`,
it will say something like this:

```
$ git remote -v
origin	https://github.com//rust-lang/rust (fetch)
origin	https://github.com//rust-lang/rust (push)
personal	https://github.com/jyn514/rust (fetch)
personal	https://github.com/jyn514/rust (push)
```

If you renamed your fork, you can change the URL like this:

```console
git remote set-url personal <URL>
```

where the `<URL>` is your new fork.

### I see 'Untracked Files: src/stdarch'?

This is left over from the move to the `library/` directory.
Unfortunately, `git rebase` does not follow renames for submodules, so you
have to delete the directory yourself:

```console
rm -r src/stdarch
```

### I see `<<< HEAD`?

You were probably in the middle of a rebase or merge conflict. See
[Conflicts](#conflicts) for how to fix the conflict. If you don't care about the changes
and just want to get a clean copy of the repository back, you can use `git reset`:

```console
# WARNING: this throws out any local changes you've made! Consider resolving the conflicts instead.
git reset --hard master
```

### Quick note about submodules

When updating your local repository with `git pull`, you may notice that sometimes
Git says you have modified some files that you have never edited. For example,
running `git status` gives you something like (note the `new commits` mention):

```
On branch master
Your branch is up to date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   src/tools/cargo (new commits)
	modified:   src/tools/rls (new commits)
	modified:   src/tools/rustfmt (new commits)

no changes added to commit (use "git add" and/or "git commit -a")
```

These changes are not changes to files: they are changes to submodules (more on
this [later](#git-submodules)). To get rid of those, run `git submodule update`
(or run any `x.py` command, which will automatically update the submodules).
Note that there is (as of <!-- date: 2021-01 --> January 2021) a bug if you use
worktrees, submodules, and x.py in a commit hook.  If you run into an error
like:

```
error: failed to read `/home/joshua/rustc-worktree/src/tools/miri/cargo-miri/Cargo.toml`

Caused by:
  No such file or directory (os error 2)
```
it's not anything you did wrong. There is a workaround at [#77620].

[#77620]: https://github.com/rust-lang/rust/issues/77620#issuecomment-705228229

## Conflicts

When you edit your code locally, you are making changes to the version of
rust-lang/rust that existed when you created your feature branch. As such, when
you submit your PR it is possible that some of the changes that have been made
to rust-lang/rust since then are in conflict with the changes you've made.

When this happens, you need to resolve the conflicts before your changes can be
merged. First, get a local copy of the conflicting changes: Checkout your local
master branch with `git checkout master`, then `git pull upstream master` to
update it with the most recent changes.

### Rebasing

You're now ready to start the rebasing process. Checkout the branch with your
changes and execute `git rebase master`.

When you rebase a branch on master, all the changes on your branch are
reapplied to the most recent version of master. In other words, Git tries to
pretend that the changes you made to the old version of master were instead
made to the new version of master. During this process, you should expect to
encounter at least one "rebase conflict." This happens when Git's attempt to
reapply the changes fails because your changes conflicted with other changes
that have been made. You can tell that this happened because you'll see
lines in the output that look like

```
CONFLICT (content): Merge conflict in file.rs
```

When you open these files, you'll see sections of the form

```
<<<<<<< HEAD
Original code
=======
Your code
>>>>>>> 8fbf656... Commit fixes 12345
```

This represents the lines in the file that Git could not figure out how to
rebase. The section between `<<<<<<< HEAD` and `=======` has the code from
master, while the other side has your version of the code. You'll need to
decide how to deal with the conflict. You may want to keep your changes,
keep the changes on master, or combine the two.

Generally, resolving the conflict consists of two steps: First, fix the
particular conflict. Edit the file to make the changes you want and remove the
`<<<<<<<`, `=======` and `>>>>>>>` lines in the process. Second, check the
surrounding code. If there was a conflict, its likely there are some logical
errors lying around too! It's a good idea to run `x.py check` here to make sure
there are no glaring errors.

Once you're all done fixing the conflicts, you need to stage the files that had
conflicts in them via `git add`. Afterwards, run `git rebase --continue` to let
Git know that you've resolved the conflicts and it should finish the rebase.
Once the rebase has succeeded, you'll want to update the associated branch on
your fork with `git push --force-with-lease`.

Note that `git push` will not work properly and say something like this:

```
 ! [rejected]        issue-xxxxx -> issue-xxxxx (non-fast-forward)
error: failed to push some refs to 'https://github.com/username/rust.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

The advice this gives is incorrect! Because of Rust's
["no-merge" policy](#no-merge-policy) the merge commit created by `git pull`
will not be allowed in the final PR, in addition to defeating the point of the
rebase! Use `git push --force-with-lease` instead.

## Advanced Rebasing

If your branch contains multiple consecutive rewrites of the same code, or if
the rebase conflicts are extremely severe, you can use
`git rebase --interactive master` to gain more control over the process. This
allows you to choose to skip commits, edit the commits that you do not skip,
change the order in which they are applied, or "squash" them into each other.

Alternatively, you can sacrifice the commit history like this:

```
# squash all the changes into one commit so you only have to worry about conflicts once
git rebase -i $(git merge-base master HEAD)  # and squash all changes along the way
git rebase master
# fix all merge conflicts
git rebase --continue
```

"Squashing" commits into each other causes them to be merged into a single
commit. Both the upside and downside of this is that it simplifies the history.
On the one hand, you lose track of the steps in which changes were made, but
the history becomes easier to work with.

You also may want to squash just the last few commits together, possibly
because they only represent "fixups" and not real changes. For example,
`git rebase --interactive HEAD~2` will allow you to edit the two commits only.

## No-Merge Policy

The rust-lang/rust repo uses what is known as a "rebase workflow." This means
that merge commits in PRs are not accepted. As a result, if you are running
`git merge` locally, chances are good that you should be rebasing instead. Of
course, this is not always true; if your merge will just be a fast-forward,
like the merges that `git pull` usually performs, then no merge commit is
created and you have nothing to worry about. Running `git config merge.ff only`
once will ensure that all the merges you perform are of this type, so that you
cannot make a mistake.

There are a number of reasons for this decision and like all others, it is a
tradeoff. The main advantage is the generally linear commit history. This
greatly simplifies bisecting and makes the history and commit log much easier
to follow and understand.

## Git submodules

**NOTE**: submodules are a nice thing to know about, but it *isn't* an absolute
prerequisite to contribute to `rustc`. If you are using Git for the first time,
you might want to get used to the main concepts of Git before reading this section.

The `rust-lang/rust` repository uses [Git submodules] as a way to use other
Rust projects from within the `rust` repo. Examples include Rust's fork of
`llvm-project` and many devtools such as `cargo`, `rust-analyzer` and `rustfmt`.

Those projects are developped and maintained in an separate Git (and GitHub)
repository, and they have their own Git history/commits, issue tracker and PRs.
Submodules allow us to create some sort of embedded sub-repository inside the
`rust` repository and use them like they were directories in the `rust` repository.

Take `miri` for example. `miri` is maintained in the [`rust-lang/miri`] repository,
but it is used in `rust-lang/rust` by the compiler for const evaluation. We bring it
in `rust` as a submodule, in the `src/tools/miri` folder.

The contents of submodules are ignored by Git: submodules are in some sense isolated
from the rest of the repository. However, if you try to `cd src/tools/miri` and then
run `git status`:

```
HEAD detached at 3fafb835
nothing to commit, working tree clean
```

As far as git is concerned, you are no longer in the `rust` repo, but in the `miri` repo.
You will notice that we are in "detatched HEAD" state, i.e. not on a branch but on a
particular commit.

This is because, like any dependency, we want to be able to control which version to use.
Submodules allow us to do just that: every submodule is "pinned" to a certain
commit, which doesn't change unless modified manually. If you use `git checkout <commit>`
in the `miri` directory and go back to the `rust` directory, you can stage this
change like any other. This is usually done by the maintainers of the
project, and looks like [this][miri-update].

Git submodules take some time to get used to, so don't worry if it isn't perfectly
clear yet. You will rarely have to use them directly and, again, you don't need
to know everything about submodules to contribute to Rust. Just know that they
exist and that they correspond to some sort of embedded subrepository dependency
that Git can nicely and fairly conveniently handle for us.

[Git submodules]: https://git-scm.com/book/en/v2/Git-Tools-Submodules
[`rust-toolstate`]: https://rust-lang-nursery.github.io/rust-toolstate/
[`rust-lang/miri`]: https://github.com/rust-lang/miri
[miri-update]: https://github.com/rust-lang/rust/pull/77500/files
