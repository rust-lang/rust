# Using External Repositories

The `rust-lang/rust` git repository depends on several other repos in the `rust-lang` organization.
There are three main ways we use dependencies:
1. As a Cargo dependency through crates.io (e.g. `rustc-rayon`)
2. As a git (e.g. `clippy`) or a [josh] (e.g. `miri`) subtree
3. As a git submodule (e.g. `cargo`)

As a general rule:
- Use crates.io for libraries that could be useful for others in the ecosystem
- Use subtrees for tools that depend on compiler internals and need to be updated if there are breaking
changes
- Use submodules for tools that are independent of the compiler

## External Dependencies (subtrees)

The following external projects are managed using some form of a `subtree`:

* [clippy](https://github.com/rust-lang/rust-clippy)
* [miri](https://github.com/rust-lang/miri)
* [rustfmt](https://github.com/rust-lang/rustfmt)
* [rust-analyzer](https://github.com/rust-lang/rust-analyzer)
* [rustc_codegen_cranelift](https://github.com/rust-lang/rustc_codegen_cranelift)

In contrast to `submodule` dependencies
(see below for those), the `subtree` dependencies are just regular files and directories which can
be updated in tree. However, if possible, enhancements, bug fixes, etc. specific
to these tools should be filed against the tools directly in their respective
upstream repositories. The exception is that when rustc changes are required to
implement a new tool feature or test, that should happen in one collective rustc PR.

`subtree` dependencies are currently managed by two distinct approaches:

* Using `git subtree`
  * `clippy` ([sync guide](https://doc.rust-lang.org/nightly/clippy/development/infrastructure/sync.html#performing-the-sync-from-rust-langrust-to-clippy))
  * `rustfmt`
  * `rustc_codegen_cranelift` ([sync script](https://github.com/rust-lang/rustc_codegen_cranelift/blob/113af154d459e41b3dc2c5d7d878e3d3a8f33c69/scripts/rustup.sh#L7))
* Using the [josh] tool
  * `miri` ([sync guide](https://github.com/rust-lang/miri/blob/master/CONTRIBUTING.md#advanced-topic-syncing-with-the-rustc-repo))
  * `rust-analyzer` ([sync script](https://github.com/rust-lang/rust-analyzer/blob/2e13684be123eca7181aa48e043e185d8044a84a/xtask/src/release.rs#L147))

The [josh] tool is an alternative to git subtrees, which manages git history in a different way and scales better to larger repositories. Specific tooling is required to work with josh, you can check out the `miri` or `rust-analyzer` scripts linked above for inspiration. If you want to migrate a subtree from `git subtree` to josh, you can check out [this guide](https://hackmd.io/7pOuxnkdQDaL1Y1FQr65xg).

Below you can find a guide on how to perform push and pull synchronization with the main rustc repo using `git subtree`, although these instructions might differ repo from repo.

### Synchronizing a subtree

Periodically the changes made to subtree based dependencies need to be synchronized between this
repository and the upstream tool repositories.

Subtree synchronizations are typically handled by the respective tool maintainers. Other users
are welcome to submit synchronization PRs, however, in order to do so you will need to modify
your local git installation and follow a very precise set of instructions.
These instructions are documented, along with several useful tips and tricks, in the
[syncing subtree changes][clippy-sync-docs] section in Clippy's Contributing guide.
The instructions are applicable for use with any subtree based tool, just be sure to
use the correct corresponding subtree directory and remote repository.

The synchronization process goes in two directions: `subtree push` and `subtree pull`.

A `subtree push` takes all the changes that happened to the copy in this repo and creates commits
on the remote repo that match the local changes. Every local
commit that touched the subtree causes a commit on the remote repo, but
is modified to move the files from the specified directory to the tool repo root.

A `subtree pull` takes all changes since the last `subtree pull`
from the tool repo and adds these commits to the rustc repo along with a merge commit that moves
the tool changes into the specified directory in the Rust repository.

It is recommended that you always do a push first and get that merged to the tool master branch.
Then, when you do a pull, the merge works without conflicts.
While it's definitely possible to resolve conflicts during a pull, you may have to redo the conflict
resolution if your PR doesn't get merged fast enough and there are new conflicts. Do not try to
rebase the result of a `git subtree pull`, rebasing merge commits is a bad idea in general.

You always need to specify the `-P` prefix to the subtree directory and the corresponding remote
repository. If you specify the wrong directory or repository
you'll get very fun merges that try to push the wrong directory to the wrong remote repository.
Luckily you can just abort this without any consequences by throwing away either the pulled commits
in rustc or the pushed branch on the remote and try again. It is usually fairly obvious
that this is happening because you suddenly get thousands of commits that want to be synchronized.

[clippy-sync-docs]: https://doc.rust-lang.org/nightly/clippy/development/infrastructure/sync.html

### Creating a new subtree dependency

If you want to create a new subtree dependency from an existing repository, call (from this
repository's root directory!)

```
git subtree add -P src/tools/clippy https://github.com/rust-lang/rust-clippy.git master
```

This will create a new commit, which you may not rebase under any circumstances! Delete the commit
and redo the operation if you need to rebase.

Now you're done, the `src/tools/clippy` directory behaves as if Clippy were
part of the rustc monorepo, so no one but you (or others that synchronize
subtrees) actually needs to use `git subtree`.

## External Dependencies (submodules)

Building Rust will also use external git repositories tracked using [git
submodules]. The complete list may be found in the [`.gitmodules`] file. Some
of these projects are required (like `stdarch` for the standard library) and
some of them are optional (like `src/doc/book`).

Usage of submodules is discussed more in the [Using Git chapter](git.md#git-submodules).

Some of the submodules are allowed to be in a "broken" state where they
either don't build or their tests don't pass, e.g. the documentation books
like [The Rust Reference]. Maintainers of these projects will be notified
when the project is in a broken state, and they should fix them as soon
as possible. The current status is tracked on the [toolstate website].
More information may be found on the Forge [Toolstate chapter].
In practice, it is very rare for documentation to have broken toolstate.

Breakage is not allowed in the beta and stable channels, and must be addressed
before the PR is merged. They are also not allowed to be broken on master in
the week leading up to the beta cut.

[git submodules]: https://git-scm.com/book/en/v2/Git-Tools-Submodules
[`.gitmodules`]: https://github.com/rust-lang/rust/blob/master/.gitmodules
[The Rust Reference]: https://github.com/rust-lang/reference/
[toolstate website]: https://rust-lang-nursery.github.io/rust-toolstate/
[Toolstate chapter]: https://forge.rust-lang.org/infra/toolstate.html
[josh]: https://josh-project.github.io/josh/intro.html
