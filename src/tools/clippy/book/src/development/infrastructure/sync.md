# Syncing changes between Clippy and [`rust-lang/rust`]

Clippy currently gets built with a pinned nightly version.

In the `rust-lang/rust` repository, where rustc resides, there's a copy of
Clippy that compiler hackers modify from time to time to adapt to changes in the
unstable API of the compiler.

We need to sync these changes back to this repository periodically, and the
changes made to this repository in the meantime also need to be synced to the
`rust-lang/rust` repository.

To avoid flooding the `rust-lang/rust` PR queue, this two-way sync process is
done in a bi-weekly basis if there's no urgent changes. This is done starting on
the day of the Rust stable release and then every other week. That way we
guarantee that we keep this repo up to date with the latest compiler API, and
every feature in Clippy is available for 2 weeks in nightly, before it can get
to beta. For reference, the first sync following this cadence was performed the
2020-08-27.

This process is described in detail in the following sections. For general
information about `subtree`s in the Rust repository see [the rustc-dev-guide][subtree].

[subtree]: https://rustc-dev-guide.rust-lang.org/external-repos.html#external-dependencies-subtree

## Patching git-subtree to work with big repos

Currently, there's a bug in `git-subtree` that prevents it from working properly
with the [`rust-lang/rust`] repo. There's an open PR to fix that, but it's
stale. Before continuing with the following steps, we need to manually apply
that fix to our local copy of `git-subtree`.

You can get the patched version of `git-subtree` from [here][gitgitgadget-pr].
Put this file under `/usr/lib/git-core` (making a backup of the previous file)
and make sure it has the proper permissions:

```bash
sudo cp --backup /path/to/patched/git-subtree.sh /usr/lib/git-core/git-subtree
sudo chmod --reference=/usr/lib/git-core/git-subtree~ /usr/lib/git-core/git-subtree
sudo chown --reference=/usr/lib/git-core/git-subtree~ /usr/lib/git-core/git-subtree
```

> _Note:_ The first time running `git subtree push` a cache has to be built.
> This involves going through the complete Clippy history once. For this you
> have to increase the stack limit though, which you can do with `ulimit -s
> 60000`. Make sure to run the `ulimit` command from the same session you call
> git subtree.

> _Note:_ If you are a Debian user, `dash` is the shell used by default for
> scripts instead of `sh`. This shell has a hardcoded recursion limit set to
> 1,000. In order to make this process work, you need to force the script to run
> `bash` instead. You can do this by editing the first line of the `git-subtree`
> script and changing `sh` to `bash`.

> Note: The following sections assume that you have set up remotes following the
> instructions in [defining remotes].

[gitgitgadget-pr]: https://github.com/gitgitgadget/git/pull/493
[defining remotes]: release.md#defining-remotes

## Performing the sync from [`rust-lang/rust`] to Clippy

Here is a TL;DR version of the sync process (all the following commands have
to be run inside the `rust` directory):

1. Clone the [`rust-lang/rust`] repository or make sure it is up-to-date.
2. Checkout the commit from the latest available nightly. You can get it using
   `rustup check`.
3. Sync the changes to the rust-copy of Clippy to your Clippy fork:
    ```bash
    # Be sure to either use a net-new branch, e.g. `rustup`, or delete the branch beforehand
    # because changes cannot be fast forwarded and you have to run this command again.
    git subtree push -P src/tools/clippy clippy-local rustup
    ```

    > _Note:_ Most of the time you have to create a merge commit in the
    > `rust-clippy` repo (this has to be done in the Clippy repo, not in the
    > rust-copy of Clippy):
    ```bash
    git fetch upstream  # assuming upstream is the rust-lang/rust remote
    git switch rustup
    git merge upstream/master --no-ff
    ```
    > Note: This is one of the few instances where a merge commit is allowed in
    > a PR.
4. Bump the nightly version in the Clippy repository by running these commands:
   ```bash
   cargo dev sync update_nightly
   git commit -m "Bump nightly version -> YYYY-MM-DD" rust-toolchain.toml clippy_utils/README.md
   ```
5. Open a PR to `rust-lang/rust-clippy` and wait for it to get merged (to
   accelerate the process ping the `@rust-lang/clippy` team in your PR and/or
   ask them in the [Zulip] stream.)

[Zulip]: https://rust-lang.zulipchat.com/#narrow/stream/clippy
[`rust-lang/rust`]: https://github.com/rust-lang/rust

## Performing the sync from Clippy to [`rust-lang/rust`]

All the following commands have to be run inside the `rust` directory.

1. Make sure you have checked out the latest `master` of `rust-lang/rust`.
2. Sync the `rust-lang/rust-clippy` master to the rust-copy of Clippy:
    ```bash
    git switch -c clippy-subtree-update
    git subtree pull -P src/tools/clippy clippy-upstream master
    ```
3. Open a PR to [`rust-lang/rust`]
