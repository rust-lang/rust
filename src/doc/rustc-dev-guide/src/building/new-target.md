# Adding a new target

These are a set of steps to add support for a new target. There are
numerous end states and paths to get there, so not all sections may be
relevant to your desired goal.

See also the associated documentation in the [target tier policy].

<!-- toc -->

[target tier policy]: https://doc.rust-lang.org/rustc/target-tier-policy.html#adding-a-new-target

## Specifying a new LLVM

For very new targets, you may need to use a different fork of LLVM
than what is currently shipped with Rust. In that case, navigate to
the `src/llvm-project` git submodule (you might need to run `./x
check` at least once so the submodule is updated), check out the
appropriate commit for your fork, then commit that new submodule
reference in the main Rust repository.

An example would be:

```
cd src/llvm-project
git remote add my-target-llvm some-llvm-repository
git checkout my-target-llvm/my-branch
cd ..
git add llvm-project
git commit -m 'Use my custom LLVM'
```

### Using pre-built LLVM

If you have a local LLVM checkout that is already built, you may be
able to configure Rust to treat your build as the system LLVM to avoid
redundant builds.

You can tell Rust to use a pre-built version of LLVM using the `target` section
of `bootstrap.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
llvm-config = "/path/to/llvm/llvm-7.0.1/bin/llvm-config"
```

If you are attempting to use a system LLVM, we have observed the following paths
before, though they may be different from your system:

- `/usr/bin/llvm-config-8`
- `/usr/lib/llvm-8/bin/llvm-config`

Note that you need to have the LLVM `FileCheck` tool installed, which is used
for codegen tests. This tool is normally built with LLVM, but if you use your
own preinstalled LLVM, you will need to provide `FileCheck` in some other way.
On Debian-based systems, you can install the `llvm-N-tools` package (where `N`
is the LLVM version number, e.g. `llvm-8-tools`). Alternately, you can specify
the path to `FileCheck` with the `llvm-filecheck` config item in `bootstrap.toml`
or you can disable codegen test with the `codegen-tests` item in `bootstrap.toml`.

## Creating a target specification

You should start with a target JSON file. You can see the specification
for an existing target using `--print target-spec-json`:

```
rustc -Z unstable-options --target=wasm32-unknown-unknown --print target-spec-json
```

Save that JSON to a file and modify it as appropriate for your target.

### Adding a target specification

Once you have filled out a JSON specification and been able to compile
somewhat successfully, you can copy the specification into the
compiler itself.

You will need to add a line to the big table inside of the
`supported_targets` macro in the `rustc_target::spec` module. You
will then add a corresponding file for your new target containing a
`target` function.

Look for existing targets to use as examples.

After adding your target to the `rustc_target` crate you may want to add
`core`, `std`, ... with support for your new target. In that case you will
probably need access to some `target_*` cfg. Unfortunately when building with
stage0 (a precompiled compiler), you'll get an error that the target cfg is
unexpected because stage0 doesn't know about the new target specification and
we pass `--check-cfg` in order to tell it to check.

To fix the errors you will need to manually add the unexpected value to the
different `Cargo.toml` in `library/{std,alloc,core}/Cargo.toml`. Here is an
example for adding `NEW_TARGET_ARCH` as `target_arch`:

*`library/std/Cargo.toml`*:
```diff
  [lints.rust.unexpected_cfgs]
  level = "warn"
  check-cfg = [
      'cfg(bootstrap)',
-      'cfg(target_arch, values("xtensa"))',
+      # #[cfg(bootstrap)] NEW_TARGET_ARCH
+      'cfg(target_arch, values("xtensa", "NEW_TARGET_ARCH"))',
```

To use this target in bootstrap, we need to explicitly add the target triple to the `STAGE0_MISSING_TARGETS`
list in `src/bootstrap/src/core/sanity.rs`. This is necessary because the default compiler bootstrap uses does
not recognize the new target we just added. Therefore, it should be added to `STAGE0_MISSING_TARGETS` so that the
bootstrap is aware that this target is not yet supported by the stage0 compiler.

```diff
const STAGE0_MISSING_TARGETS: &[&str] = &[
+   "NEW_TARGET_TRIPLE"
];
```

## Patching crates

You may need to make changes to crates that the compiler depends on,
such as [`libc`][] or [`cc`][]. If so, you can use Cargo's
[`[patch]`][patch] ability. For example, if you want to use an
unreleased version of `libc`, you can add it to the top-level
`Cargo.toml` file:

```diff
diff --git a/Cargo.toml b/Cargo.toml
index 1e83f05e0ca..4d0172071c1 100644
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -113,6 +113,8 @@ cargo-util = { path = "src/tools/cargo/crates/cargo-util" }
 [patch.crates-io]
+libc = { git = "https://github.com/rust-lang/libc", rev = "0bf7ce340699dcbacabdf5f16a242d2219a49ee0" }

 # See comments in `src/tools/rustc-workspace-hack/README.md` for what's going on
 # here
 rustc-workspace-hack = { path = 'src/tools/rustc-workspace-hack' }
```

After this, run `cargo update -p libc` to update the lockfiles.

Beware that if you patch to a local `path` dependency, this will enable
warnings for that dependency. Some dependencies are not warning-free, and due
to the `deny-warnings` setting in `bootstrap.toml`, the build may suddenly start
to fail.
To work around warnings, you may want to:
- Modify the dependency to remove the warnings
- Or for local development purposes, suppress the warnings by setting deny-warnings = false in bootstrap.toml.

```toml
# bootstrap.toml
[rust]
deny-warnings = false
```

[`libc`]: https://crates.io/crates/libc
[`cc`]: https://crates.io/crates/cc
[patch]: https://doc.rust-lang.org/stable/cargo/reference/overriding-dependencies.html#the-patch-section

## Cross-compiling

Once you have a target specification in JSON and in the code, you can
cross-compile `rustc`:

```
DESTDIR=/path/to/install/in \
./x install -i --stage 1 --host aarch64-apple-darwin.json --target aarch64-apple-darwin \
compiler/rustc library/std
```

If your target specification is already available in the bootstrap
compiler, you can use it instead of the JSON file for both arguments.

## Promoting a target from tier 2 (target) to tier 2 (host)

There are two levels of tier 2 targets:
  a) Targets that are only cross-compiled (`rustup target add`)
  b) Targets that [have a native toolchain][tier2-native] (`rustup toolchain install`)

[tier2-native]: https://doc.rust-lang.org/nightly/rustc/target-tier-policy.html#tier-2-with-host-tools

For an example of promoting a target from cross-compiled to native,
see [#75914](https://github.com/rust-lang/rust/pull/75914).
