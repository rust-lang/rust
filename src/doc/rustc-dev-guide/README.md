[![CI](https://github.com/rust-lang/rustc-dev-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-lang/rustc-dev-guide/actions/workflows/ci.yml)

This is a collaborative effort to build a guide that explains how rustc works.
The aim of the guide is to help new contributors get oriented to rustc,
as well as to help more experienced folks in figuring out
some new part of the compiler that they haven't worked on before.

You may also find the [rustc API docs] useful.

Note that these are not intended as a guide; it's recommended that you search
for the docs you're looking for instead of reading them top to bottom.

For documentation on developing the standard library, see
[`std-dev-guide`](https://std-dev-guide.rust-lang.org/).

### Contributing to the guide

The guide is useful today, but it has a lot of work still to go.

If you'd like to help improve the guide, we'd love to have you!
You can find plenty of issues on the [issue
tracker](https://github.com/rust-lang/rustc-dev-guide/issues).
Just post a comment on the issue you would like to work on to make sure that we don't
accidentally duplicate work.
If you think something is missing, please open an issue about it!

**In general, if you don't know how the compiler works, that is not a
problem!** In that case, what we will do is to schedule a bit of time
for you to talk with someone who **does** know the code, or who wants
to pair with you and figure it out.
Then you can work on writing up what you learned.

In general, when writing about a particular part of the compiler's code, we
recommend that you link to the relevant parts of the [rustc API docs].

The guide has a much lower bar for what it takes for a PR to be merged.
Check out the forge documentation for [our policy][forge_policy].

[forge_policy]: https://forge.rust-lang.org/rustc-dev-guide/index.html#review-policy

### Build Instructions

To build a local static HTML site, install [`mdbook`](https://github.com/rust-lang/mdBook) with:

```
cargo install mdbook mdbook-linkcheck2 mdbook-mermaid
```

and execute the following command in the root of the repository:

```
mdbook build --open
```

The build files are found in the `book/html` directory.

### Link Validations

We use `mdbook-linkcheck2` to validate URLs included in our documentation.
Link checking is **not** run by default locally, though it is in CI.
To enable it locally, set the environment variable `ENABLE_LINKCHECK=1` like in the
following example.

```
ENABLE_LINKCHECK=1 mdbook serve
```

## Synchronizing josh subtree with rustc

This repository is linked to `rust-lang/rust` as a [josh](https://josh-project.github.io/josh/intro.html) subtree.
You can use the [rustc-josh-sync](https://github.com/rust-lang/josh-sync) tool to perform synchronization.

You can find a guide on how to perform the synchronization [here](./src/external-repos.md#synchronizing-a-josh-subtree).

[rustc API docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle
