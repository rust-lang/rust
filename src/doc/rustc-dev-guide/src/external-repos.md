# Using external repositories

The `rust-lang/rust` git repository depends on several other repos in the `rust-lang` organization.
There are three main ways we use dependencies:
1. As a Cargo dependency through crates.io (e.g. `rustc-rayon`)
2. As a [josh] (e.g. `miri`) subtree
3. As a git submodule (e.g. `cargo`)

As a general rule:
- Use crates.io for libraries that could be useful for others in the ecosystem
- Use subtrees for tools that depend on compiler internals and need to be updated if there are breaking
  changes
- Use submodules for tools that are independent of the compiler

## External dependencies (subtrees)

The following external projects are managed using a Josh `subtree`:

* [clippy](https://github.com/rust-lang/rust-clippy)
* [miri](https://github.com/rust-lang/miri)
* [portable-simd](https://github.com/rust-lang/portable-simd)
* [rustfmt](https://github.com/rust-lang/rustfmt)
* [rust-analyzer](https://github.com/rust-lang/rust-analyzer)
* [rustc_codegen_cranelift](https://github.com/rust-lang/rustc_codegen_cranelift)
* [rustc_codegen_gcc](https://github.com/rust-lang/rustc_codegen_gcc)
* [rustc-dev-guide](https://github.com/rust-lang/rustc-dev-guide)
* [compiler-builtins](https://github.com/rust-lang/compiler-builtins)
* [stdarch](https://github.com/rust-lang/stdarch)

In contrast to `submodule` dependencies (see below for those),
the `subtree` dependencies are just regular files and directories which can be updated in-tree.
However, if possible, enhancements, bug fixes, etc. specific
to these tools should be filed against the tools directly in their respective upstream repositories.
The exception is that when rustc changes are required to implement a new tool feature or test,
that should happen in one collective rustc PR.

Similarly, if your rustc changes break the build of some subtree dependency,
you should include fixes for those in the rustc PR as well.

### Josh subtrees

The [josh] tool is an alternative to git subtrees, which manages git history in a different way and 
scales better to larger repositories. We used to manage subtrees using `git subtree`, but it was
very slow and difficult to use. Josh works much better.

Subtrees are used for bidirectional synchronization between the rustc repo (`rust-lang/rust`) and 
the subtree repo (e.g. Clippy). There are two operations that are regularly performed on 
subtrees:

- A subtree pull takes all changes made to the subtree directory in the rustc repo, and re-applies 
  them in the subtree repo.
- A subtree push takes all changes made to the subtree repo, and re-applies them to the subtree 
  directory in the rustc repo.

### Synchronizing a Josh subtree

Specific tooling is required to work with Josh. We use a dedicated tool called 
[`rustc-josh-sync`][josh-sync] to make Josh subtree updates easier.

The commands below can be used for all our Josh subtrees, although note that `miri`
requires you to perform some [additional steps](https://github.com/rust-lang/miri/blob/master/CONTRIBUTING.md#advanced-topic-syncing-with-the-rustc-repo) during pulls.

You can install the tool using the following command:
```
cargo install --locked --git https://github.com/rust-lang/josh-sync
```

Both pulls and pushes are performed from the subtree repository. So first
switch to the subtree repository checkout directory in your terminal before performing a pull or a 
push.

#### Performing pull
1. Checkout a new branch that will be used to create a PR into the subtree
    - Ideally, the branch should start at the latest commit of the default branch of the subtree.
2. Run the pull command:
    ```bash
    rustc-josh-sync pull
    ```
3. Push the branch to your fork and create a PR into the subtree repository
    - If you have the `gh` CLI installed, `rustc-josh-sync` can create the PR for you.

#### Performing push

> NOTE:
> Before you proceed, look at some guidance related to Git [on josh-sync README].

1. Checkout the latest commit of the default branch of the subtree.
2. Run the push command to create a branch named `<branch-name>` in a `rustc` fork under the 
   `<gh-username>` account
    ```
    rustc-josh-sync push <branch-name> <gh-username>
    ```
3. Create a PR from `<branch-name>` into `rust-lang/rust`
    - If you have the `gh` CLI installed, `rustc-josh-sync` can create the PR for you.

### Creating a new Josh subtree dependency

If you want to migrate a repository dependency from a `git submodule` to josh, you 
can check out [this guide](https://hackmd.io/7pOuxnkdQDaL1Y1FQr65xg). If you want to add a new Josh 
subtree that was not a submodule before, follow the guide as if you migrated a 
submodule to a Josh subtree.

## External dependencies (submodules)

Building Rust will also use external git repositories tracked using [git submodules].
The complete list may be found in the [`.gitmodules`] file.
Some of these projects are required (like `stdarch` for the standard library) and
some of them are optional (like `src/doc/book`).

Usage of submodules is discussed more in the [Using Git chapter](git.md#git-submodules).

Some of the submodules are allowed to be in a "broken" state where they
either don't build or their tests don't pass, e.g. the documentation books
like [The Rust Reference].
Maintainers of these projects will be notified when the project is in a broken state,
and they should fix them as soon as possible.
The current status is tracked on the [toolstate website].
More information may be found on the Forge [Toolstate chapter].
In practice, it is very rare for documentation to have broken toolstate.

Breakage is not allowed in the beta and stable channels,
and must be addressed before the PR is merged.
They are also not allowed to be broken on `main` in the week leading up to the beta cut.

[git submodules]: https://git-scm.com/book/en/v2/Git-Tools-Submodules
[`.gitmodules`]: https://github.com/rust-lang/rust/blob/HEAD/.gitmodules
[The Rust Reference]: https://github.com/rust-lang/reference/
[toolstate website]: https://rust-lang-nursery.github.io/rust-toolstate/
[Toolstate chapter]: https://forge.rust-lang.org/infra/toolstate.html
[josh]: https://josh-project.github.io/josh/intro.html
[josh-sync]: https://github.com/rust-lang/josh-sync
[on josh-sync README]: https://github.com/rust-lang/josh-sync#git-peculiarities
