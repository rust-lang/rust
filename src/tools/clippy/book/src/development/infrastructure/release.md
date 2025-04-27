# Release a new Clippy Version

> _NOTE:_ This document is probably only relevant to you, if you're a member of
> the Clippy team.

Clippy is released together with stable Rust releases. The dates for these
releases can be found at the [Rust Forge]. This document explains the necessary
steps to create a Clippy release.

1. [Defining Remotes](#defining-remotes)
1. [Bump Version](#bump-version)
1. [Find the Clippy commit](#find-the-clippy-commit)
1. [Update the `beta` branch](#update-the-beta-branch)
1. [Update the `stable` branch](#update-the-stable-branch)
1. [Tag the stable commit](#tag-the-stable-commit)
1. [Update `CHANGELOG.md`](#update-changelogmd)

[Rust Forge]: https://forge.rust-lang.org/

## Defining Remotes

You may want to define the `upstream` remote of the Clippy project to simplify
the following steps. However, this is optional and you can replace `upstream`
with the full URL instead.

```bash
git remote add upstream git@github.com:rust-lang/rust-clippy
```

## Bump Version

When a release needs to be done, `cargo test` will fail, if the versions in the
`Cargo.toml` are not correct. During that sync, the versions need to be bumped.
This is done by running:

```bash
cargo dev release bump_version
```

This will increase the version number of each relevant `Cargo.toml` file. After
that, just commit the updated files with:

```bash
git commit -m "Bump Clippy version -> 0.1.XY" **/*Cargo.toml
```

`XY` should be exchanged with the corresponding version

## Find the Clippy commit

For both updating the `beta` and the `stable` branch, the first step is to find
the Clippy commit of the last Clippy sync done in the respective Rust branch.

Running the following commands _in the Rust repo_ will get the commit for the
specified `<branch>`:

```bash
git switch <branch>
SHA=$(git log --oneline -- src/tools/clippy/ | grep -o "Merge commit '[a-f0-9]*' into .*" | head -1 | sed -e "s/Merge commit '\([a-f0-9]*\)' into .*/\1/g")
```

Where `<branch>` is one of `stable`, `beta`, or `master`.

## Update the `beta` branch

After getting the commit of the `beta` branch, the `beta` branch in the Clippy
repository can be updated.

```bash
git checkout beta
git reset --hard $SHA
git push upstream beta
```

## Update the `stable` branch

After getting the commit of the `stable` branch, the `stable` branch in the
Clippy repository can be updated.

```bash
git checkout stable
git reset --hard $SHA
git push upstream stable
```

## Tag the `stable` commit

After updating the `stable` branch, tag the HEAD commit and push it to the
Clippy repo.

```bash
git tag rust-1.XX.0               # XX should be exchanged with the corresponding version
git push upstream rust-1.XX.0     # `upstream` is the `rust-lang/rust-clippy` remote
```

After this, the release should be available on the Clippy [tags page].

[tags page]: https://github.com/rust-lang/rust-clippy/tags

## Publish `clippy_utils`

The `clippy_utils` crate is published to `crates.io` without any stability
guarantees. To do this, after the [sync] and the release is done, switch back to
the `upstream/master` branch and publish `clippy_utils`:

> Note: The Rustup PR bumping the nightly and Clippy version **must** be merged
> before doing this.

```bash
git switch master && git pull upstream master
cargo publish --manifest-path clippy_utils/Cargo.toml
```

[sync]: sync.md

## Update `CHANGELOG.md`

For this see the document on [how to update the changelog].

If you don't have time to do a complete changelog update right away, just update
the following parts:

- Remove the `(beta)` from the new stable version:

  ```markdown
  ## Rust 1.XX (beta) -> ## Rust 1.XX
  ```

- Update the release date line of the new stable version:

  ```markdown
  Current beta, release 20YY-MM-DD -> Current stable, released 20YY-MM-DD
  ```

- Update the release date line of the previous stable version:

  ```markdown
  Current stable, released 20YY-MM-DD -> Released 20YY-MM-DD
  ```

[how to update the changelog]: changelog_update.md
