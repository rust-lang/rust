# Release a new Clippy Version

_NOTE: This document is probably only relevant to you, if you're a member of the
Clippy team._

Clippy is released together with stable Rust releases. The dates for these
releases can be found at the [Rust Forge]. This document explains the necessary
steps to create a Clippy release.

1. [Find the Clippy commit](#find-the-clippy-commit)
2. [Tag the stable commit](#tag-the-stable-commit)
3. [Update `CHANGELOG.md`](#update-changelogmd)
4. [Remerge the `beta` branch](#remerge-the-beta-branch)
5. [Update the `beta` branch](#update-the-beta-branch)

_NOTE: This document is for stable Rust releases, not for point releases. For
point releases, step 1. and 2. should be enough._

[Rust Forge]: https://forge.rust-lang.org/


## Find the Clippy commit

The first step is to tag the Clippy commit, that is included in the stable Rust
release. This commit can be found in the Rust repository.

```bash
# Assuming the current directory corresponds to the Rust repository
$ git fetch upstream    # `upstream` is the `rust-lang/rust` remote
$ git checkout 1.XX.0   # XX should be exchanged with the corresponding version
$ git submodule update
$ SHA=$(git submodule status src/tools/clippy | awk '{print $1}')
```


## Tag the stable commit

After finding the Clippy commit, it can be tagged with the release number.

```bash
# Assuming the current directory corresponds to the Clippy repository
$ git checkout $SHA
$ git tag rust-1.XX.0               # XX should be exchanged with the corresponding version
$ git push upstream master --tags   # `upstream` is the `rust-lang/rust-clippy` remote
```

After this, the release should be available on the Clippy [release page].

[release page]: https://github.com/rust-lang/rust-clippy/releases


## Update `CHANGELOG.md`

For this see the document on [how to update the changelog].

[how to update the changelog]: https://github.com/rust-lang/rust-clippy/blob/master/doc/changelog_update.md


## Remerge the `beta` branch

This step is only necessary, if since the last release something was backported
to the beta Rust release. The remerge is then necessary, to make sure that the
Clippy commit, that was used by the now stable Rust release, persists in the
tree of the Clippy repository.

To find out if this step is necessary run

```bash
# Assumes that the local master branch is up-to-date
$ git fetch upstream
$ git branch master --contains upstream/beta
```

If this command outputs `master`, this step is **not** necessary.

```bash
# Assuming `HEAD` is the current `master` branch of rust-lang/rust-clippy
$ git checkout -b backport_remerge
$ git merge beta
$ git diff  # This diff has to be empty, otherwise something with the remerge failed
$ git push origin backport_remerge  # This can be pushed to your fork
```

After this, open a PR to the master branch. In this PR, the commit hash of the
`HEAD` of the `beta` branch must exists. In addition to that, no files should
be changed by this PR.


## Update the `beta` branch

This step must be done **after** the PR of the previous step was merged.

First, the Clippy commit of the `beta` branch of the Rust repository has to be
determined.

```bash
# Assuming the current directory corresponds to the Rust repository
$ git checkout beta
$ git submodule update
$ BETA_SHA=$(git submodule status src/tools/clippy | awk '{print $1}')
```

After finding the Clippy commit, the `beta` branch in the Clippy repository can
be updated.

```bash
# Assuming the current directory corresponds to the Clippy repository
$ git checkout beta
$ git rebase $BETA_SHA
$ git push upstream beta
```
