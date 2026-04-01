# `rustfmt` subtree sync procedure

Note that `rustfmt` has not migrated to `josh` yet, so the git subtree sync is somewhat involved.
This procedure is mostly adapted from the `clippy` subtree sync process at
<https://doc.rust-lang.org/stable/clippy/development/infrastructure/sync.html>, but adapted for
`rustfmt`. We are keeping a separate copy of the instructions for `rustfmt` in case `clippy` moves
off of `git subtree` in the mean time (and also slightly adjusted to be `rustfmt`-specific.

> [!NOTE]
>
> Note that AFAIK, eventually both `clippy` and `rustfmt` would like to move to the `josh`-sync
> workflow, just that `rustfmt` is blocked on actually doing a bidirectional sync first, while
> `clippy` is sorting out some issues related to tags and git history (due to usage of `git
> subtree`).

## Tooling

The `git-subtree` tooling still has a bug that prevents it from working properly with the
`rust-lang/rust` repository. This means that you need to build and use a patched version of
`git-subtree`.

> [!NOTE]
>
> The patched version of `git-subtree` is the sources corresponding to the stale PR
> <https://github.com/gitgitgadget/git/pull/493>.

On Linux, place `git-subtree` under `/usr/lib/git-core` (make sure to keep a backup copy of the
'standard' `git-subtree`), and make sure it has proper permissions:

```sh
$ sudo cp --backup /path/to/patched/git-subtree.sh /usr/lib/git-core/git-subtree
$ sudo chmod --reference=/usr/lib/git-core/git-subtree~ /usr/lib/git-core/git-subtree
$ sudo chown --reference=/usr/lib/git-core/git-subtree~ /usr/lib/git-core/git-subtree
```

> [!NOTE]
>
> Running `git subtree push` for the first time requires building a cache, which involves going
> through the entire history of `rustfmt` once. You likely will need to increase the stack limit via
>
> ```sh
> $ ulimit -s 60000
> ```

> [!NOTE]
>
> The following steps assume that you have configured the `rustfmt` remote as `upstream`, i.e.
>
> ```sh
> $ git remote add upstream git@github.com:rust-lang/rustfmt
> ```

## Subtree push direction: syncing changes from `rust-lang/rust` to `rustfmt`

> [!WARNING]
>
> **For this subtree-push direction, all commands described must be run within the `rust-lang/rust`
> checkout.**

### 1. Acquire a checkout of `rust-lang/rust`

Either acquire a clone of `rust-lang/rust`, or if you already have a checkout, make sure the
checkout is up-to-date via `git fetch`.

### 2. Checkout the commit from the latest available nightly

 You can fetch the commit hash of the latest available nightly by inspecting `rustup check` output.

### 3. Sync changes from the `rust`-copy of `rustfmt` to your `rustfmt` fork

> [!WARNING]
>
> **Make sure to either use a fresh branch, e.g. `subtree-push`, or delete the branch beforehand**.
> Changes cannot be fast forwarded and you have to run this command again.

```sh
$ git subtree push -P src/tools/rustfmt /path/to/rustfmt/checkout subtree-push
```

Most of the time, you will need to create a **merge commit** in the `rustfmt` repository. Note that
this must be done in the subtree repo (i.e. `rustfmt` repo) and not in the `rust`-copy of `rustfmt`.

Assuming the `upstream` remote is the `rust-lang/rust` remote:

```sh
$ git fetch upstream
$ git switch subtree-push
$ git merge upstream/main --no-ff
```

> [!WARNING]
>
> You may have to manually resolve certain merge conflicts. Pay extra attention when resolving them,
> since it's easy to accidentally resolve the conflict in incorrect ways.

> [!TIP]
>
> Subtree syncs are one of the rare occasions where a merge commit is allowed in a PR.

### 4. Bump the nightly toolchain version in the `rustfmt` repository

Using the same latest nightly date (that you can obtain by inspecting `rustup check` output),
manually edit `rust-toolchain`:

```diff
 [toolchain]
-channel = "nightly-2025-04-02"
+channel = "nightly-$LATEST_NIGHTLY_DATE"
 components = ["llvm-tools", "rustc-dev"]
```

Substituting `$LATEST_NIGHTLY_DATE` with the latest nightly date.

Create a separate commit dedicated to making the `rust-toolchain` change. You can use [the following
commit message template](#rust-toolchain-bump-commit-message-template).

#### `rust-toolchain` bump commit message template

```text
chore: bump rustfmt toolchain to nightly-$LATEST_NIGHTLY_DATE

Bumping the toolchain version as part of a git subtree push.

current toolchain (nightly-$CURRENT_NIGHTLY_DATE):
   - $CURRENT_NIGHTLY_VERSION-nightly ($CURRENT_NIGHTLY_HASH $CURRENT_NIGHTLY_DATE)

latest toolchain (nightly-$LATEST_NIGHTLY_DATE):
   - $LATEST_NIGHTLY_VERSION-nightly ($LATEST_NIGHTLY_HASH $LATEST_NIGHTLY_DATE)
```

Substituting the placeholders with the right information.

> [!TIP]
>
> Example bump commit message:
>
> ```text
> chore: bump rustfmt toolchain to nightly-2025-10-07
>
> Bumping the toolchain version as part of a git subtree push.
>
> current toolchain (nightly-2025-04-02): - 1.88.0-nightly (e2014e876 2025-04-01)
>
> latest toolchain (nightly-2025-10-07): - 1.92.0-nightly (f6aa851db 2025-10-07)
> ```

### 5. Open a PR against `rustfmt`

And wait for the sync PR to be merged. The `rustfmt` maintainers will run Diff Check against the PR
to catch any unexpected formatting changes. Once Diff Check failures are investigated and are
resolved, the PR can then be merged.

For the PR:

- Use the title `subtree-push nightly-$LATEST_NIGHTLY_DATE` for consistency with previous
  subtree-pushes.
- Include a copy of the bump commit message in the PR description for quick reference. Feel free to
  include additional notes that might be helpful for the maintainers when reviewing.

> [!TIP]
>
> Example subtree-push PR title and description:
>
> **PR title**: `subtree-push nightly-2025-10-07`
>
> **PR description**:
>
> ```text
> Bumping the toolchain version as part of a git subtree push.
>
> current toolchain (nightly-2025-04-02):
> - 1.88.0-nightly (e2014e876 2025-04-01)
>
> latest toolchain (nightly-2025-10-07):
> - 1.92.0-nightly (f6aa851db 2025-10-07)
> ```

> [!WARNING]
>
> Make sure to immediately follow-up with a subtree-pull direction, syncing `rustfmt` to
> `rust-lang/rust`. We need the {subtree-push, subtree-pull} directions to be performed in
> lock-step, to minimize any changes in between that makes the logistics more complex.

## Subtree pull direction: syncing from `rustfmt` to `rust-lang/rust`

> [!WARNING]
>
> For this **subtree-pull** direction, all commands must also be performed within the
> `rust-lang/rust` checkout.

### 1. Make sure latest `main` of `rust-lang/rust` is checked out

### 2. Sync `rustfmt` `main` to the `rust`-copy of `rustfmt`

```sh
$ git switch -c rustfmt-subtree-update
$ git subtree pull -P src/tools/rustfmt /path/to/rustfmt/checkout main
```

### 3. Open a PR against `rust-lang/rust`

Use the PR title

> `rustfmt` subtree update

so that `triagebot` will not warn against the PR containing a merge commit, and makes it easy for
`rustfmt` maintainers to discover.

Back link to the `rustfmt` subtree-push PR as helpful context.
