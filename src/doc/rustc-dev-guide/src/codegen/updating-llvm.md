# Updating LLVM

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

There are two primary reasons nowadays that we want to update LLVM in one way or
another:

* First, a bug could have been fixed! Often we find bugs in the compiler and fix
  them upstream in LLVM. We'll want to pull fixes back to the compiler itself as
  they're merged upstream.

* Second, a new feature may be avaiable in LLVM that we want to use in rustc,
  but we don't want to wait for a full LLVM release to test it out.

Each of these reasons has a different strategy for updating LLVM, and we'll go
over both in detail here.

## Bugfix Updates

For updates of LLVM that typically just update a bug, we cherry-pick the bugfix
to the branch we're already using. The steps for this are:

1. Make sure the bugfix is in upstream LLVM.
2. Identify the branch that rustc is currently using. The `src/llvm-project`
   submodule is always pinned to a branch of the
   [rust-lang/llvm-project](https://github.com/rust-lang/llvm-project) repository.
3. Fork the rust-lang/llvm-project repository
4. Check out the appropriate branch (typically named `rustc/a.b-yyyy-mm-dd`)
5. Cherry-pick the upstream commit onto the branch
6. Push this branch to your fork
7. Send a Pull Request to rust-lang/llvm-project to the same branch as before
8. Wait for the PR to be merged
9. Send a PR to rust-lang/rust updating the `src/llvm-project` submodule with
   your bugfix
10. Wait for PR to be merged

The tl;dr; is that we can cherry-pick bugfixes at any time and pull them back
into the rust-lang/llvm-project branch that we're using, and getting it into the
compiler is just updating the submodule via a PR!

Example PRs look like:
[#59089](https://github.com/rust-lang/rust/pull/59089)

## Feature updates

> Note that this is all information as applies to the current day in age. This
> process for updating LLVM changes with practically all LLVM updates, so this
> may be out of date!

Unlike bugfixes, updating to pick up a new feature of LLVM typically requires a
lot more work. This is where we can't reasonably cherry-pick commits backwards
so we need to do a full update. There's a lot of stuff to do here, so let's go
through each in detail.

1. Create a new branch in the rust-lang/llvm-project repository. This branch
   should be named `rustc/a.b-yyyy-mm-dd` where `a.b` is the current version
   number of LLVM in-tree at the time of the branch and the remaining part is
   today's date.

2. Apply Rust-specific patches to the llvm-project repository. All features and
   bugfixes are upstream, but there's often some weird build-related patches
   that don't make sense to upstream which we have on our repositories. These
   patches are around the latest patches in the rust-lang/llvm-project branch
   that rustc is currently using.

3. Update the `compiler-rt` submodule in the
   `rust-lang-nursery/compiler-builtins` repository. Push this update to the
   same branch name of the `llvm-project` submodule to the
   of the `rust-lang/compiler-rt` repository. Then push this update to a branch
   of `compiler-builtins` with the same-named branch. Note that this step is
   frequently optional since we may not need to update `compiler-rt`.

4. Prepare a commit to rust-lang/rust

  * Update `src/llvm-project`
  * Update `compiler-builtins` crate in `Cargo.lock` (if necessary)

5. Build your commit. Make sure you've committed the previous changes to ensure
   submodule updates aren't reverted. Some commands you should execute are:

   * `./x.py build src/llvm` - test that LLVM still builds
   * `./x.py build src/tools/lld` - same for LLD
   * `./x.py build` - build the rest of rustc

   You'll likely need to update `src/rustllvm/*.cpp` to compile with updated
   LLVM bindings. Note that you should use `#ifdef` and such to ensure that the
   bindings still compile on older LLVM versions.

6. Test for regressions across other platforms. LLVM often has at least one bug
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

7. Send a PR! Hopefully it's smooth sailing from here :).

For prior art, previous LLVM updates look like
[#55835](https://github.com/rust-lang/rust/pull/55835)
[#47828](https://github.com/rust-lang/rust/pull/47828)

### Caveats and gotchas

Ideally the above instructions are pretty smooth, but here's some caveats to
keep in mind while going through them:

* LLVM bugs are hard to find, don't hesitate to ask for help! Bisection is
  definitely your friend here (yes LLVM takes forever to build, yet bisection is
  still your friend)
* Updating LLDB has some Rust-specific patches currently that aren't upstream.
  If you have difficulty @tromey can likely help out.
* If you've got general questions, @alexcrichton can help you out.
* Creating branches is a privileged operation on GitHub, so you'll need someone
  with write access to create the branches for you most likely.
