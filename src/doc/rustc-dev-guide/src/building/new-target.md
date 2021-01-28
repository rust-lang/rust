# Adding a new target

These are a set of steps to add support for a new target. There are
numerous end states and paths to get there, so not all sections may be
relevant to your desired goal.

## Specifying a new LLVM

For very new targets, you may need to use a different fork of LLVM
than what is currently shipped with Rust. In that case, navigate to
the `src/llvm-project` git submodule (you might need to run `x.py
check` at least once so the submodule is updated), check out the
appropriate commit for your fork, then commit that new submodule
reference in the main Rust repository.

An example would be:

```
cd src/llvm-project
git remote add my-target-llvm some-llvm-repository
git checkout my-target-llvm/my-branch
cd ..
git add llvm_target
git commit -m 'Use my custom LLVM'
```

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

Look for existing targets to use as examples

## Patching crates

You may need to make changes to crates that the compiler depends on,
such as [`libc`][] or [`cc`][]. If so, you can use Cargo's
[`[patch]`][patch] ability. For example, if you want to use an
unreleased version of `libc`, you can add it to the top-level
`Cargo.toml` file:

```diff
diff --git a/Cargo.toml b/Cargo.toml
index be15e50e2bc..4fb1248ba99 100644
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -66,10 +66,11 @@ cargo = { path = "src/tools/cargo" }
 [patch.crates-io]
 # Similar to Cargo above we want the RLS to use a vendored version of `rustfmt`
 # that we're shipping as well (to ensure that the rustfmt in RLS and the
 # `rustfmt` executable are the same exact version).
 rustfmt-nightly = { path = "src/tools/rustfmt" }
+libc = { git = "https://github.com/rust-lang/libc", rev = "0bf7ce340699dcbacabdf5f16a242d2219a49ee0" }

 # See comments in `src/tools/rustc-workspace-hack/README.md` for what's going on
 # here
 rustc-workspace-hack = { path = 'src/tools/rustc-workspace-hack' }
```

After this, run `cargo update -p libc` to update the lockfiles.

[`libc`]: https://crates.io/crates/libc
[`cc`]: https://crates.io/crates/cc
[patch]: https://doc.rust-lang.org/stable/cargo/reference/overriding-dependencies.html#the-patch-section

## Cross-compiling

Once you have a target specification in JSON and in the code, you can
cross-compile `rustc`:

```
DESTDIR=/path/to/install/in \
./x.py install -i --stage 1 --host aarch64-apple-darwin.json --target aarch64-apple-darwin \
compiler/rustc library/std
```

If your target specification is already available in the bootstrap
compiler, you can use it instead of the JSON file for both arguments.

## Promoting a target from tier 2 (target) to tier 2 (host)

There are two levels of tier 2 targets:
a) Targets that are only cross-compiled (`rustup target add`)
b) Targets that have a native toolchain (`rustup toolchain install`)

For an example of promoting a target from cross-compiled to native,
see [#75914](https://github.com/rust-lang/rust/pull/75914).
