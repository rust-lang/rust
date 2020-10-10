[![Travis CI badge](https://api.travis-ci.com/rust-lang/rustc-dev-guide.svg?branch=master)](https://travis-ci.com/github/rust-lang/rustc-dev-guide)


This is a collaborative effort to build a guide that explains how rustc
works. The aim of the guide is to help new contributors get oriented
to rustc, as well as to help more experienced folks in figuring out
some new part of the compiler that they haven't worked on before.

[You can read the latest version of the guide here.](https://rustc-dev-guide.rust-lang.org/)

You may also find the rustdocs [for the compiler itself][rustdocs] useful.

[rustdocs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/

### Contributing to the guide

The guide is useful today, but it has a lot of work still go.

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

Deactivate the CI testing link validations by commenting out the `[output.linkcheck]` field in the `book.toml` configuration file like this:

```toml
[book]
title = "Guide to Rustc Development"
author = "Rustc developers"
description = "A guide to developing rustc"

[build]
create-missing = false

[output.html]
git-repository-url = "https://github.com/rust-lang/rustc-dev-guide"

[output.html.fold]
enable = true
level = 1

# [output.linkcheck]
# follow-web-links = false
# exclude = [ "crates\\.io", "gcc\\.godbolt\\.org", "youtube\\.com", "youtu\\.be", "dl\\.acm\\.org", "cs\\.bgu\\.ac\\.il" ]
# cache-timeout = 172800
# warning-policy = "error"
```

These validations can cause errors during local builds (see Link Validations section below).  Please **do not** commit these `book.toml` file changes when you submit a pull request.

To build a local static HTML site, install [`mdbook`](https://github.com/rust-lang/mdBook) with:

```
> cargo install mdbook
```

and execute the following command in the root of the repository:

```
> mdbook build
```

The build files are found in the `book` directory.

### Pre-commit script

We also test that line lengths are less than 100 columns. To test this locally,
you can run `ci/check_line_lengths.sh`.

You can also set this to run automatically.

On Linux:

```bash
ln -s ../../ci/check_line_lengths.sh .git/hooks/pre-commit
```

On Windows:

```powershell
New-Item -Path .git/hooks/pre-commit -ItemType HardLink -Value <absolute_path/to/check_line_lengths.sh>
```

### Link Validations

We use `mdbook-linkcheck` to validate URLs included in our documentation. To perform link checks, uncomment the `[output.linkcheck]` field in the `book.toml` configuration file and install `mdbook-linkcheck` with:

```bash
> cargo install mdbook-linkcheck --git https://github.com/Michael-F-Bryan/mdbook-linkcheck --rev 14441d77646d58cea8ffc32fde9ea33b2bedb1a2
```

Note that we use an alpha version of `mdbook-linkcheck` to be able to use a feature that hasn't landed in a release yet.
You will also need `mdbook` version `>= 0.3.5`.
`linkcheck` will be run automatically when you build with the instructions in the section above.

**Please note**: You may receive errors like the following when link checks are active on local `mdbook` builds:

```
error: The server responded with 429 Too Many Requests for "https://github.com/rust-lang/rust/tree/master/src/tools/compiletest"

   ┌── tests/intro.md:6:1 ───
   │
 6 │ [`src/tools/compiletest`] directory). This section gives a brief
   │ ^ Server responded with 429 Too Many Requests
```

There is not a workaround for this error at the moment.  Comment out the `[output.linkcheck]` field in the `book.toml` using the build instructions above to complete a local site build without link validations.


## How to fix toolstate failures

> **NOTE**: Currently, we do not track the rustc-dev-guide toolstate due to
[the spurious failure](https://github.com/rust-lang/rust/pull/71731),
but we leave this instructions for when we do it again in the future.

1. You will get a ping from the toolstate commit. e.g. https://github.com/rust-lang-nursery/rust-toolstate/commit/8ffa0e4c30ac9ba8546b7046e5c4ccc2b96ebdd4

2. The commit contains a link to the PR that caused the breakage. e.g. https://github.com/rust-lang/rust/pull/64321

3. If you go to that PR's thread, there is a post from bors with a link to the CI status: https://github.com/rust-lang/rust/pull/64321#issuecomment-529763807

4. Follow the check-actions link to get to the Actions page for that build

5. There will be approximately 1 billion different jobs for the build. They are for different configurations and platforms. The rustc-dev-guide build only runs on the Linux x86_64-gnu-tools job. So click on that job in the list, which is about 60% down in the list.

6. Click the Run build step in the job to get the console log for the step.

7. Click on the log and Ctrl-f to get a search box in the log

8. Search for rustc-dev-guide. This gets you to the place where the links are checked. It is usually ~11K lines into the log

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
./x.py test --incremental src/doc/rustc-dev-guide # This is optional and should succeed anyway
# Open a PR in rust-lang/rust
```

12. Wait for PR to merge

Voilà!
