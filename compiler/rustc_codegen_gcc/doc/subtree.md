# Subtree sync procedure

`rustc_codegen_gcc` is using [Josh] to perform subtree synces with the `rust-lang/rust` repository.

[Josh]: https://josh-project.dev/docs/intro.html

The procedure for performing subtree synchronizations is documented [here][rdg-josh].

[rdg-josh]: https://rustc-dev-guide.rust-lang.org/external-repos.html#synchronizing-a-josh-subtree

## Pushing changes to `rust-lang/rust`

If the used GCC commit (recorded in `libgccjit.version`) changed in `rustc_codegen_gcc`, then you will have to update the `gcc` submodule in the `rust-lang/rust` repository in the push sync PR. The CI in `rust-lang/rust` should ensure that these two commit references match. You can simply update the `gcc` submodule commit and push it to your push sync PR once it has been created.

## Environment variables to make things easier

To avoid passing through github `https` interface, create a file named `josh.gitconfig` (for example at `$HOME/.config/git/josh.gitconfig`) which contains:

```gitconfig
[url "git@github.com:"]
insteadOf = "https://github.com/"

[include]
path = ~/.gitconfig
path = ~/.config/git/config
```

Then refer to it using the `GIT_CONFIG_GLOBAL` environment variable.

And to prevent cloning the rust repository every time, set the `RUSTC_GIT` environment variable to point to your local copy of the `rust` repository. The path needs to be absolute!

Then running the command looks like:

```bash
GIT_CONFIG_GLOBAL=$HOME/.config/git/josh.gitconfig RUSTC_GIT=/my/local/rust-fork rustc-josh-sync push <branch name> <rust fork owner>
```

So if your github account name is "tadam" and you want to push to a branch named "cg_gcc-subtree-sync", the command will be:

```bash
GIT_CONFIG_GLOBAL=$HOME/.config/git/josh.gitconfig RUSTC_GIT=/my/local/rust-fork rustc-josh-sync push cg_gcc-subtree-sync tadam
```
