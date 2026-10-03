# `rustfmt` subtree sync procedure

`rustfmt` is using [Josh] to perform subtree synces with the `rust-lang/rust` repository.

[Josh]: https://josh-project.dev/docs/intro.html

## Tooling

The [`josh-sync`][josh-sync] tool is used as a wrapper on top of Josh to help automate the subtree
sync process (both pulls and pushes). You can install it with `cargo install --git https://github.com/rust-lang/josh-sync`.

[josh-sync]: https://github.com/rust-lang/josh-sync

## Subtree pull direction: syncing changes from `rust-lang/rust` to `rustfmt`

1. Be in a `rustfmt` checkout
2. Checkout the latest `main` branch from upstream
3. Create a new branch that will be used for the sync
4. Run `rustc-josh-sync pull`
5. Confirm the creation of a PR directly by `rustc-josh-sync`, or create a PR to `rust-lang/rustfmt` repository

The `rustfmt` maintainers will run Diff Check against the PR to catch any unexpected formatting changes.

- Maintainers should trigger Diff-Check for the combinations of Edition {2021, 2024} x Style
  Edition {2021, 2024}.

Once Diff Check failures are investigated and are resolved, the PR can then be merged.

### (Where applicable) Update changelog and bump rustfmt version number

Where applicable, we may need to update the CHANGELOG entries with merged PRs (both in `rustfmt`
repository and also in the `rust-lang/rust` `rustfmt` subtree that was included in the subtree-push
merge), and then bump rustfmt version number.

## Subtree push direction: syncing changes from `rustfmt` to rust-lang/rust`

1. Be in a `rustfmt` checkout
2. Checkout the latest `main` branch from upstream
3. Run `rustc-josh-sync push <branch-name> <github-username>`
   - Josh will push the changes to a branch with the given name in your `<github-username>/rust` fork of `rust-lang/rust`.
4. Follow the prompt of `rustc-josh-sync` to open a PR to `rust-lang/rust`, or create the PR manually.

**Make sure to minimize the time between a pull and a subsequent push to avoid unnecessary complications.**
