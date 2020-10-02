# Using Git

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
