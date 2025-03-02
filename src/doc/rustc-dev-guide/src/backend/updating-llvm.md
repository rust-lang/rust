# Updating LLVM

<!-- toc -->

<!-- date-check: Aug 2024 -->
Rust supports building against multiple LLVM versions:

* Tip-of-tree for the current LLVM development branch is usually supported
  within a few days. PRs for such fixes are tagged with `llvm-main`.
* The latest released major version is always supported.
* The one or two preceding major versions are usually supported.

By default, Rust uses its own fork in the [rust-lang/llvm-project repository].
This fork is based on a `release/$N.x` branch of the upstream project, where
`$N` is either the latest released major version, or the current major version
in release candidate phase. The fork is never based on the `main` development
branch.

Our LLVM fork only accepts:

* Backports of changes that have already landed upstream.
* Workarounds for build issues affecting our CI environment.

With the exception of one grandfathered-in patch for SGX enablement, we do not
accept functional patches that have not been upstreamed first.

There are three types of LLVM updates, with different procedures:

* Backports while the current major LLVM version is supported.
* Backports while the current major LLVM version is no longer supported (or
  the change is not eligible for upstream backport).
* Update to a new major LLVM version.

## Backports (upstream supported)

While the current major LLVM version is supported upstream, fixes should be
backported upstream first, and the release branch then merged back into the
Rust fork.

1. Make sure the bugfix is in upstream LLVM.
2. If this hasn't happened already, request a backport to the upstream release
   branch. If you have LLVM commit access, follow the [backport process].
   Otherwise, open an issue requesting the backport. Continue once the
   backport has been approved and merged.
3. Identify the branch that rustc is currently using. The `src/llvm-project`
   submodule is always pinned to a branch of the
   [rust-lang/llvm-project repository].
4. Fork the rust-lang/llvm-project repository.
5. Check out the appropriate branch (typically named `rustc/a.b-yyyy-mm-dd`).
6. Add a remote for the upstream repository using
   `git remote add upstream https://github.com/llvm/llvm-project.git` and
   fetch it using `git fetch upstream`.
7. Merge the `upstream/release/$N.x` branch.
8. Push this branch to your fork.
9. Send a Pull Request to rust-lang/llvm-project to the same branch as before.
   Be sure to reference the Rust and/or LLVM issue that you're fixing in the PR
   description.
10. Wait for the PR to be merged.
11. Send a PR to rust-lang/rust updating the `src/llvm-project` submodule with
    your bugfix. This can be done locally with `git submodule update --remote
    src/llvm-project` typically.
12. Wait for PR to be merged.

An example PR:
[#59089](https://github.com/rust-lang/rust/pull/59089)

## Backports (upstream not supported)

Upstream LLVM releases are only supported for two to three months after the
GA release. Once upstream backports are no longer accepted, changes should be
cherry-picked directly to our fork.

1. Make sure the bugfix is in upstream LLVM.
2. Identify the branch that rustc is currently using. The `src/llvm-project`
   submodule is always pinned to a branch of the
   [rust-lang/llvm-project repository].
3. Fork the rust-lang/llvm-project repository.
4. Check out the appropriate branch (typically named `rustc/a.b-yyyy-mm-dd`).
5. Add a remote for the upstream repository using
   `git remote add upstream https://github.com/llvm/llvm-project.git` and
   fetch it using `git fetch upstream`.
6. Cherry-pick the relevant commit(s) using `git cherry-pick -x`.
7. Push this branch to your fork.
8. Send a Pull Request to rust-lang/llvm-project to the same branch as before.
   Be sure to reference the Rust and/or LLVM issue that you're fixing in the PR
   description.
9. Wait for the PR to be merged.
10. Send a PR to rust-lang/rust updating the `src/llvm-project` submodule with
    your bugfix. This can be done locally with `git submodule update --remote
    src/llvm-project` typically.
11. Wait for PR to be merged.

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

   * `./x build src/llvm-project` - test that LLVM still builds
   * `./x build` - build the rest of rustc

   You'll likely need to update [`llvm-wrapper/*.cpp`][`llvm-wrapper`]
   to compile with updated LLVM bindings.
   Note that you should use `#ifdef` and such to ensure
   that the bindings still compile on older LLVM versions.

   Note that `profile = "compiler"` and other defaults set by `./x setup`
   download LLVM from CI instead of building it from source.
   You should disable this temporarily to make sure your changes are being used.
   This is done by having the following setting in `bootstrap.toml`:

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

   <!-- date-check: Sep 2024 -->
   > For prior art, here are some previous LLVM updates:
   > - [LLVM 11](https://github.com/rust-lang/rust/pull/73526)
   > - [LLVM 12](https://github.com/rust-lang/rust/pull/81451)
   > - [LLVM 13](https://github.com/rust-lang/rust/pull/87570)
   > - [LLVM 14](https://github.com/rust-lang/rust/pull/93577)
   > - [LLVM 15](https://github.com/rust-lang/rust/pull/99464)
   > - [LLVM 16](https://github.com/rust-lang/rust/pull/109474)
   > - [LLVM 17](https://github.com/rust-lang/rust/pull/115959)
   > - [LLVM 18](https://github.com/rust-lang/rust/pull/120055)
   > - [LLVM 19](https://github.com/rust-lang/rust/pull/127513)

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
[backport process]: https://llvm.org/docs/GitHub.html#backporting-fixes-to-the-release-branches
