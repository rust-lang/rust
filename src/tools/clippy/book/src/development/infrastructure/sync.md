# Syncing changes between Clippy and [`rust-lang/rust`]

Clippy currently gets built with a pinned nightly version.

In the `rust-lang/rust` repository, where rustc resides, there's a copy of Clippy that compiler devs modify from time to
time to adapt to changes in the unstable API of the compiler.

We need to sync these changes back to this repository periodically, and the changes made to this repository in the
meantime also need to be synced to the `rust-lang/rust` repository.

To avoid flooding the `rust-lang/rust` PR queue, this two-way sync process is done in a bi-weekly basis if there's no
urgent changes. This is done starting on the day of the Rust stable release and then every other week. That way we
guarantee that we keep this repo up to date with the latest compiler API, and every feature in Clippy is available for 2
weeks in nightly, before it can get to beta. For reference, the first sync following this cadence was performed on
2020-08-27.

This process is described in detail in the following sections.

## Installing `rustc-josh-sync`

The sync is done with [JOSH] through the [`rustc-josh-sync`] tool. Install this tool with the following command:

```sh
cargo install --locked --git https://github.com/rust-lang/josh-sync
```

[JOSH]: https://josh-project.github.io/josh/
[`rustc-josh-sync`]: https://github.com/rust-lang/josh-sync

## Performing the sync from [`rust-lang/rust`] to Clippy

First, checkout a new branch called `rustup` on top of the latest `main` branch:

```
git switch -c rustup upstream/main
```

To do the sync, run:

```
rustc-josh-sync pull
```

This command will update the nightly toolchain in the `rust-toolchain` file and will pull the changes from the Rust
repository.

If there should be merge conflicts `rustc-josh-sync` will prompt you to resolve the conflict. After resolving them and
run `git merge --continue`. Then, respond to the prompt from `rustc-josh-sync` with `y` to continue the sync process.

Finally, make sure `cargo test --features=internal` passes and potentially commit any necessary changes.

> [!NOTE]
> If the version tests fail, refer to [bump version] in the release documentation.

If there were no merge conflicts, the tool will ask you if it should create a PR with `gh`. Either do that or create a
PR manually.

> [!NOTE]
> If you are a Clippy maintainer, you can add `r? @ghost` to the PR description and merge the PR yourself, after a
> quick sanity review.

[bump version]: release.md#bump-version

## Performing the sync from Clippy to [`rust-lang/rust`]

The other direction is done by running

```
rustc-josh-sync push clippy-subtree-update <GitHub-name>
```

Where the `<GitHub-name>` is your GitHub user name. This is required for pushing the sync to GitHub and opening a PR.

> [!NOTE]
> By default, rustc-josh-sync will create a new `rustc-checkout` dir and clones the `rust-lang/rust` repository
> into it. If you want to use an existing checkout, you can prefix the command with `RUSTC_GIT=/path/to/rust`.

If everything went right, there will be a GitHub link that has to be used to open the sync PR in the Rust repository.
The PR description should look something like this:

```
rust-clippy subtree update

Subtree update of `rust-clippy` to https://github.com/rust-lang/rust-clippy/commit/{head}.

Created using https://github.com/rust-lang/josh-sync.

r? @ghost
```

The title must be kept as is, to [tell triagebot] that this is a sync PR.

The first line of the body must be kept as is, to [find the Clippy commit] during a release.

Change `r? @ghost` to the GitHub handle of a Clippy maintainer, so that they can review and approve the PR.

[find the Clippy commit]: release.md#find-the-clippy-commit
[tell triagebot]: https://github.com/rust-lang/rust/pull/114157

[`rust-lang/rust`]: https://github.com/rust-lang/rust
