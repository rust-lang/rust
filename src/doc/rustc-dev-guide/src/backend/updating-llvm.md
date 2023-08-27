# Updating LLVM

<!-- toc -->

<!-- date-check: Jul 2023 -->
There is no formal policy about when to update LLVM or what it can be updated to,
but a few guidelines are applied:

* We try to always support the latest released version
* We try to support the last few versions
  (and the number changes over time)
* We allow moving to arbitrary commits during development
* We strongly prefer to upstream all patches to LLVM before including them in rustc

## Why update LLVM?

There are two reasons we would want to update LLVM:

* A bug could have been fixed!
  Note that if we are the ones who fixed such a bug,
  we prefer to upstream it, then pull it back for use by rustc.

* LLVM itself may have a new release.

Each of these reasons has a different strategy for updating LLVM, and we'll go
over them in detail here.

## Bugfix Updates

For updates of LLVM that are to fix a small bug, we cherry-pick the bugfix to
the branch we're already using. The steps for this are:

1. Make sure the bugfix is in upstream LLVM.
2. Identify the branch that rustc is currently using. The `src/llvm-project`
   submodule is always pinned to a branch of the
   [rust-lang/llvm-project repository].
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

An example PR:
[#59089](https://github.com/rust-lang/rust/pull/59089)

## New LLVM Release Updates

<!-- date-check: Jul 2023 -->

Unlike bugfixes,
updating to a new release of LLVM typically requires a lot more work.
This is where we can't reasonably cherry-pick commits backwards,
so we need to do a full update.
There's a lot of stuff to do here,
so let's go through each in detail.

1. LLVM announces that its latest release version has branched.
   This will show up as a branch in the [llvm/llvm-project repository],
   typically named `release/$N.x`,
   where `$N` is the version of LLVM that's being released.

1. Create a new branch in the [rust-lang/llvm-project repository]
   from this `release/$N.x` branch,
   and name it `rustc/a.b-yyyy-mm-dd`,
   where `a.b` is the current version number of LLVM in-tree
   at the time of the branch,
   and the remaining part is the current date.

2. Apply Rust-specific patches to the llvm-project repository.
   All features and bugfixes are upstream,
   but there's often some weird build-related patches
   that don't make sense to upstream.
   These patches are typically the latest patches in the
   rust-lang/llvm-project branch that rustc is currently using.

3. Build the new LLVM in the `rust` repository.
   To do this,
   you'll want to update the `src/llvm-project` repository to your branch,
   and the revision you've created.
   It's also typically a good idea to update `.gitmodules` with the new
   branch name of the LLVM submodule.
   Make sure you've committed changes to
   `src/llvm-project` to ensure submodule updates aren't reverted.
   Some commands you should execute are:

   * `./x build src/llvm` - test that LLVM still builds
   * `./x build src/tools/lld` - same for LLD
   * `./x build` - build the rest of rustc

   You'll likely need to update [`llvm-wrapper/*.cpp`][`llvm-wrapper`]
   to compile with updated LLVM bindings.
   Note that you should use `#ifdef` and such to ensure
   that the bindings still compile on older LLVM versions.

   Note that `profile = "compiler"` and other defaults set by `./x setup`
   download LLVM from CI instead of building it from source.
   You should disable this temporarily to make sure your changes are being used.
   This is done by having the following setting in `config.toml`:

   ```toml
   [llvm]
   download-ci-llvm = false
   ```

4. Test for regressions across other platforms. LLVM often has at least one bug
   for non-tier-1 architectures, so it's good to do some more testing before
   sending this to bors! If you're low on resources you can send the PR as-is
   now to bors, though, and it'll get tested anyway.

   Ideally, build LLVM and test it on a few platforms:

   * Linux
   * macOS
   * Windows

   Afterwards, run some docker containers that CI also does:

   * `./src/ci/docker/run.sh wasm32`
   * `./src/ci/docker/run.sh arm-android`
   * `./src/ci/docker/run.sh dist-various-1`
   * `./src/ci/docker/run.sh dist-various-2`
   * `./src/ci/docker/run.sh armhf-gnu`

5. Prepare a PR to `rust-lang/rust`. Work with maintainers of
   `rust-lang/llvm-project` to get your commit in a branch of that repository,
   and then you can send a PR to `rust-lang/rust`. You'll change at least
   `src/llvm-project` and will likely also change [`llvm-wrapper`] as well.

   > For prior art, here are some previous LLVM updates:
   > - [LLVM 11](https://github.com/rust-lang/rust/pull/73526)
   > - [LLVM 12](https://github.com/rust-lang/rust/pull/81451)
   > - [LLVM 13](https://github.com/rust-lang/rust/pull/87570)
   > - [LLVM 14](https://github.com/rust-lang/rust/pull/93577)
   > - [LLVM 15](https://github.com/rust-lang/rust/pull/99464)
   > - [LLVM 16](https://github.com/rust-lang/rust/pull/109474)

   Note that sometimes it's easiest to land [`llvm-wrapper`] compatibility as a PR
   before actually updating `src/llvm-project`.
   This way,
   while you're working through LLVM issues,
   others interested in trying out the new LLVM can benefit from work you've done
   to update the C++ bindings.

3. Over the next few months,
   LLVM will continually push commits to its `release/a.b` branch.
   We will often want to have those bug fixes as well.
   The merge process for that is to use `git merge` itself to merge LLVM's
   `release/a.b` branch with the branch created in step 2.
   This is typically
   done multiple times when necessary while LLVM's release branch is baking.

4. LLVM then announces the release of version `a.b`.

5. After LLVM's official release,
   we follow the process of creating a new branch on the
   rust-lang/llvm-project repository again,
   this time with a new date.
   It is only then that the PR to update Rust to use that version is merged.

   The commit history of `rust-lang/llvm-project`
   should look much cleaner as a `git rebase` is done,
   where just a few Rust-specific commits are stacked on top of stock LLVM's release branch.

### Caveats and gotchas

Ideally the above instructions are pretty smooth, but here's some caveats to
keep in mind while going through them:

* LLVM bugs are hard to find, don't hesitate to ask for help!
  Bisection is definitely your friend here
  (yes LLVM takes forever to build, yet bisection is still your friend).
  Note that you can make use of [Dev Desktops],
  which is an initiative to provide the contributors with remote access to powerful hardware.
* If you've got general questions, [wg-llvm] can help you out.
* Creating branches is a privileged operation on GitHub, so you'll need someone
  with write access to create the branches for you most likely.


[rust-lang/llvm-project repository]: https://github.com/rust-lang/llvm-project
[llvm/llvm-project repository]: https://github.com/llvm/llvm-project
[`llvm-wrapper`]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_llvm/llvm-wrapper
[wg-llvm]: https://rust-lang.zulipchat.com/#narrow/stream/187780-t-compiler.2Fwg-llvm
[Dev Desktops]: https://forge.rust-lang.org/infra/docs/dev-desktop.html
