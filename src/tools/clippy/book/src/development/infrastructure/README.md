# Infrastructure

In order to deploy Clippy over `rustup`, some infrastructure is necessary. This
chapter describes the different parts of the Clippy infrastructure that need to
be maintained to make this possible.

The most important part is the sync between the `rust-lang/rust` repository and
the Clippy repository that takes place every two weeks. This process is
described in the [Syncing changes between Clippy and `rust-lang/rust`](sync.md)
section.

A new Clippy release is done together with every Rust release, so every six
weeks. The release process is described in the [Release a new Clippy
Version](release.md) section. During a release cycle a changelog entry for the
next release has to be written. The format of that and how to do that is
documented in the [Changelog Update](changelog_update.md) section.

> _Note:_ The Clippy CI should also be described in this chapter, but for now is
> left as a TODO.
