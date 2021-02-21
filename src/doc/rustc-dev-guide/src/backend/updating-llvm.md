# Updating LLVM

<!-- toc -->

The Rust compiler uses LLVM as its primary codegen backend today, and naturally
we want to at least occasionally update this dependency! Currently we do not
have a strict policy about when to update LLVM or what it can be updated to, but
a few guidelines are applied:

* We try to always support the latest released version of LLVM
* We try to support the "last few" versions of LLVM (how many is changing over
  time)
* We allow moving to arbitrary commits during development.
* Strongly prefer to upstream all patches to LLVM before including them in
  rustc.

This policy may change over time (or may actually start to exist as a formal
policy!), but for now these are rough guidelines!

## Why update LLVM?

There are a few reasons nowadays that we want to update LLVM in one way or
another:

* A bug could have been fixed! Often we find bugs in the compiler and fix
  them upstream in LLVM. We'll want to pull fixes back to the compiler itself as
  they're merged upstream.

* A new feature may be available in LLVM that we want to use in rustc,
  but we don't want to wait for a full LLVM release to test it out.

* LLVM itself may have a new release and we'd like to update to this LLVM
  release.

Each of these reasons has a different strategy for updating LLVM, and we'll go
over them in detail here.

## Bugfix Updates

For updates of LLVM that are to fix a small bug, we cherry-pick the bugfix to
the branch we're already using. The steps for this are:

1. Make sure the bugfix is in upstream LLVM.
2. Identify the branch that rustc is currently using. The `src/llvm-project`
   submodule is always pinned to a branch of the
   [rust-lang/llvm-project](https://github.com/rust-lang/llvm-project) repository.
3. Fork the rust-lang/llvm-project repository
4. Check out the appropriate branch (typically named `rustc/a.b-yyyy-mm-dd`)
5. Cherry-pick the upstream commit onto the branch
6. Push this branch to your fork
7. Send a Pull Request to rust-lang/llvm-project to the same branch as before.
   Be sure to reference the Rust and/or LLVM issue that you're fixing in the PR
   description.
8. Wait for the PR to be merged
9. Send a PR to rust-lang/rust updating the `src/llvm-project` submodule with
   your bugfix. This can be done locally with `git submodule update --remote
   src/llvm-project` typically.
10. Wait for PR to be merged

The tl;dr; is that we can cherry-pick bugfixes at any time and pull them back
into the rust-lang/llvm-project branch that we're using, and getting it into the
compiler is just updating the submodule via a PR!

Example PRs look like:
[#59089](https://github.com/rust-lang/rust/pull/59089)

## Feature updates

> Note that this information is as of the time of this writing <!-- date:
2018-12 --> (December 2018). The process for updating LLVM changes with
practically all LLVM updates, so this may be out of date!

Unlike bugfixes, updating to pick up a new feature of LLVM typically requires a
lot more work. This is where we can't reasonably cherry-pick commits backwards
so we need to do a full update. There's a lot of stuff to do here, so let's go
through each in detail.

1. Create a new branch in the rust-lang/llvm-project repository. This branch
   should be named `rustc/a.b-yyyy-mm-dd` where `a.b` is the current version
   number of LLVM in-tree at the time of the branch and the remaining part is
   today's date. Move this branch to the commit in LLVM that you'd like, which
   for this is probably the current LLVM HEAD.

2. Apply Rust-specific patches to the llvm-project repository. All features and
   bugfixes are upstream, but there's often some weird build-related patches
   that don't make sense to upstream which we have on our repositories. These
   patches are around the latest patches in the rust-lang/llvm-project branch
   that rustc is currently using.

3. Build the new LLVM in the `rust` repository. To do this you'll want to update
   the `src/llvm-project` repository to your branch and the revision you've
   created. It's also typically a good idea to update `.gitmodules` with the new
   branch name of the LLVM submodule. Make sure you've committed changes to
   `src/llvm-project` to ensure submodule updates aren't reverted. Some commands
   you should execute are:

   * `./x.py build src/llvm` - test that LLVM still builds
   * `./x.py build src/tools/lld` - same for LLD
   * `./x.py build` - build the rest of rustc

   You'll likely need to update [`llvm-wrapper/*.cpp`][`llvm-wrapper`] to compile
   with updated LLVM bindings. Note that you should use `#ifdef` and such to ensure
   that the bindings still compile on older LLVM versions.

   Note that `profile = "compiler"` and other defaults set by `x.py setup`
   download LLVM from CI instead of building it from source. You should
   disable this temporarily to make sure your changes are being used, by setting
   ```toml
   [llvm]
   download-ci-llvm = false
   ```
   in config.toml.

4. Test for regressions across other platforms. LLVM often has at least one bug
   for non-tier-1 architectures, so it's good to do some more testing before
   sending this to bors! If you're low on resources you can send the PR as-is
   now to bors, though, and it'll get tested anyway.

   Ideally, build LLVM and test it on a few platforms:

   * Linux
   * OSX
   * Windows

   and afterwards run some docker containers that CI also does:

   * `./src/ci/docker/run.sh wasm32-unknown`
   * `./src/ci/docker/run.sh arm-android`
   * `./src/ci/docker/run.sh dist-various-1`
   * `./src/ci/docker/run.sh dist-various-2`
   * `./src/ci/docker/run.sh armhf-gnu`

5. Prepare a PR to `rust-lang/rust`. Work with maintainers of
   `rust-lang/llvm-project` to get your commit in a branch of that repository,
   and then you can send a PR to `rust-lang/rust`. You'll change at least
   `src/llvm-project` and will likely also change [`llvm-wrapper`] as well.

For prior art, previous LLVM updates look like
[#55835](https://github.com/rust-lang/rust/pull/55835)
[#47828](https://github.com/rust-lang/rust/pull/47828)
[#62474](https://github.com/rust-lang/rust/pull/62474)
[#62592](https://github.com/rust-lang/rust/pull/62592). Note that sometimes it's
easiest to land [`llvm-wrapper`] compatibility as a PR before actually updating
`src/llvm-project`. This way while you're working through LLVM issues others
interested in trying out the new LLVM can benefit from work you've done to
update the C++ bindings.

[`llvm-wrapper`]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_llvm/llvm-wrapper

### Caveats and gotchas

Ideally the above instructions are pretty smooth, but here's some caveats to
keep in mind while going through them:

* LLVM bugs are hard to find, don't hesitate to ask for help! Bisection is
  definitely your friend here (yes LLVM takes forever to build, yet bisection is
  still your friend)
* If you've got general questions, @alexcrichton can help you out.
* Creating branches is a privileged operation on GitHub, so you'll need someone
  with write access to create the branches for you most likely.

## New LLVM Release Updates

Updating to a new release of LLVM is very similar to the "feature updates"
section above. The release process for LLVM is often months-long though and we
like to ensure compatibility ASAP. The main tweaks to the "feature updates"
section above is generally around branch naming. The sequence of events
typically looks like:

1. LLVM announces that its latest release version has branched. This will show
   up as a branch in https://github.com/llvm/llvm-project typically named
   `release/$N.x` where `$N` is the version of LLVM that's being released.

2. We then follow the "feature updates" section above to create a new branch of
   LLVM in our rust-lang/llvm-project repository. This follows the same naming
   convention of branches as usual, except that `a.b` is the new version. This
   update is eventually landed in the rust-lang/rust repository.

3. Over the next few months, LLVM will continually push commits to its
   `release/a.b` branch. Often those are bug fixes we'd like to have as well.
   The merge process for that is to use `git merge` itself to merge LLVM's
   `release/a.b` branch with the branch created in step 2. This is typically
   done multiple times when necessary while LLVM's release branch is baking.

4. LLVM then announces the release of version `a.b`.

5. After LLVM's official release, we follow the "feature update" section again
   to create a new branch in the rust-lang/llvm-project repository, this time
   with a new date. The commit history should look much cleaner as just a few
   Rust-specific commits stacked on top of stock LLVM's release branch.
