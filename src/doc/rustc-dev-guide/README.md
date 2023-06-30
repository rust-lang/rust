[![CI](https://github.com/rust-lang/rustc-dev-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-lang/rustc-dev-guide/actions/workflows/ci.yml)


This is a collaborative effort to build a guide that explains how rustc
works. The aim of the guide is to help new contributors get oriented
to rustc, as well as to help more experienced folks in figuring out
some new part of the compiler that they haven't worked on before.

[You can read the latest version of the guide here.](https://rustc-dev-guide.rust-lang.org/)

You may also find the rustdocs [for the compiler itself][rustdocs] useful.
Note that these are not intended as a guide; it's recommended that you search
for the docs you're looking for instead of reading them top to bottom.

[rustdocs]: https://doc.rust-lang.org/nightly/nightly-rustc

For documentation on developing the standard library, see
[`std-dev-guide`](https://std-dev-guide.rust-lang.org/).

### Contributing to the guide

The guide is useful today, but it has a lot of work still to go.

If you'd like to help improve the guide, we'd love to have you! You can find
plenty of issues on the [issue
tracker](https://github.com/rust-lang/rustc-dev-guide/issues). Just post a
comment on the issue you would like to work on to make sure that we don't
accidentally duplicate work. If you think something is missing, please open an
issue about it!

**In general, if you don't know how the compiler works, that is not a
problem!** In that case, what we will do is to schedule a bit of time
for you to talk with someone who **does** know the code, or who wants
to pair with you and figure it out.  Then you can work on writing up
what you learned.

In general, when writing about a particular part of the compiler's code, we
recommend that you link to the relevant parts of the [rustc
rustdocs][rustdocs].

### Build Instructions

To build a local static HTML site, install [`mdbook`](https://github.com/rust-lang/mdBook) with:

```
> cargo install mdbook mdbook-linkcheck mdbook-toc mdbook-mermaid
```

and execute the following command in the root of the repository:

```
> mdbook build --open
```

The build files are found in the `book/html` directory.

### Link Validations

We use `mdbook-linkcheck` to validate URLs included in our documentation.
`linkcheck` will be run automatically when you build with the instructions in the section above.

### Table of Contents

We use `mdbook-toc` to auto-generate TOCs for long sections. You can invoke the preprocessor by
including the `<!-- toc -->` marker at the place where you want the TOC.

### Pre-commit script

We also test that line lengths are less than 100 columns. To test this locally,
you can run `ci/lengthcheck.sh`.

You can also set this to run automatically.

On Linux:

```bash
ln -s ../../ci/lengthcheck.sh .git/hooks/pre-commit
```

On Windows:

```powershell
New-Item -Path .git/hooks/pre-commit -ItemType HardLink -Value $(Resolve-Path ci/lengthcheck.sh)
```

## How to fix toolstate failures

> **NOTE**: Currently, we do not track the rustc-dev-guide toolstate due to
[spurious failures](https://github.com/rust-lang/rust/pull/71731),
but we leave these instructions for when we do it again in the future.

1. You will get a ping from the toolstate commit. e.g. https://github.com/rust-lang-nursery/rust-toolstate/commit/8ffa0e4c30ac9ba8546b7046e5c4ccc2b96ebdd4

2. The commit contains a link to the PR that caused the breakage. e.g. https://github.com/rust-lang/rust/pull/64321

3. If you go to that PR's thread, there is a post from bors with a link to the CI status: https://github.com/rust-lang/rust/pull/64321#issuecomment-529763807

4. Follow the check-actions link to get to the Actions page for that build

5. There will be approximately 1 billion different jobs for the build. They are for different configurations and platforms. The rustc-dev-guide build only runs on the Linux x86_64-gnu-tools job. So click on that job in the list, which is about 60% down in the list.

6. Click the Run build step in the job to get the console log for the step.

7. Click on the log and Ctrl-f to get a search box in the log

8. Search for rustc-dev-guide. This gets you to the place where the links are checked. It is usually ~11K lines into the log.

9. Look at the links in the log near that point in the log

10. Fix those links in the rustc-dev-guide (by making a PR in the rustc-dev-guide repo)

11. Make a PR on the rust-lang/rust repo to update the rustc-dev-guide git submodule in src/docs/rustc-dev-guide.
To make a PR, the following steps are useful.

```bash
# Assuming you already cloned the rust-lang/rust repo and you're in the correct directory
git submodule update --remote src/doc/rustc-dev-guide
git add -u
git commit -m "Update rustc-dev-guide"
# Note that you can use -i, which is short for --incremental, in the following command
./x test --incremental src/doc/rustc-dev-guide # This is optional and should succeed anyway
# Open a PR in rust-lang/rust
```

12. Wait for PR to merge

Voilà!
