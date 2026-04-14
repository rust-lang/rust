# Rustc on ThingOS: Build Status Audit (April 2026)

## Summary

The Linux-hosted stage-1 cross-compiler bootstrap **succeeds** as of April 12,
2026.  Running `cargo xtask rustc-thingos` (or
`just rustc-thingos`) drives `x.py` through all three stages
successfully and caches the result under `target/rustc-thingos/`.

The produced compiler is a **Linux-hosted cross-compiler** — it runs on
`x86_64-unknown-linux-gnu` and can cross-compile code targeting
`x86_64-unknown-thingos`.  It is **not yet** a ThingOS-native compiler and is
therefore **not staged into the ISO** by default.

---

## Blocking issues: resolved

Three issues were tracked as blockers in issue #713.  All three have been
closed:

### #714 — LLVM platform shims (`EnvPathSeparator`, `getSize`)

**Status: resolved** via CMake patches.

The original LLVM errors occurred because the bootstrap mapped
`x86_64-unknown-thingos` to `CMAKE_SYSTEM_NAME=Generic`, causing LLVM to omit
Unix-specific declarations.

Two structured patches fix this:

| Patch | Target repo | What it does |
|---|---|---|
| `patches/rust/vendor-rust/0001-bootstrap-use-thingos-cmake-system-name.patch` | `vendor/rust` | Changes `llvm.rs` to emit `CMAKE_SYSTEM_NAME=ThingOS` and `LLVM_ON_UNIX=ON` instead of `Generic` |
| `patches/rust/llvm-project/0001-llvm-classify-thingos-as-unix.patch` | `vendor/rust/src/llvm-project` | Classifies `ThingOS` as a Unix-like platform in `config-ix.cmake` and `HandleLLVMOptions.cmake` |

These patches are applied automatically by `just rust-apply-patches`.

### #716 — Refine `llvm-target` in target JSON

**Status: resolved** without changing the JSON field.

Issue #716 asked whether `"llvm-target": "x86_64-unknown-none"` should be
changed to something more Unix-flavoured.  The CMake patch approach adopted for
#714 makes the `llvm-target` value irrelevant for the LLVM host-tool build: the
`CMAKE_SYSTEM_NAME=ThingOS` override is applied unconditionally when the target
triple contains `thingos`.

`targets/x86_64-unknown-thingos.json` retains `"llvm-target":
"x86_64-unknown-none"` to preserve the bare-metal code-generation profile for
user binaries.

### #717 — Fix rustlib staging path and re-enable rustc-thingos in ISO build

**Status: partially resolved** — wiring is in place; ISO staging remains
intentionally disabled.

- `xtask/src/image.rs` calls `stage_rustc_for_iso(sh, iso_root)` (line ~638).
- `xtask/src/main.rs` calls `rustc_thingos::build_rustc_thingos(&sh, &env)` in
  all ISO/run paths.
- `build_rustc_thingos` runs by default and respects the `SKIP_RUSTC_THINGOS=1`
  env-var opt-out gate.
- `stage_rustc_for_iso` prints an advisory message and returns `Ok(())` without
  copying anything — intentionally, because the cached binary is a Linux ELF.

When a ThingOS-native compiler is available the function body needs to be
filled in to copy `rustlib/x86_64-unknown-thingos/` (not the full rustlib tree)
and the `rustc` binary into the ISO.

---

## Patch inventory

### Applied by `just rust-apply-patches`

| File | Applies to | Description |
|---|---|---|
| `patches/rust/vendor-rust/0001-bootstrap-use-thingos-cmake-system-name.patch` | `vendor/rust` | Bootstrap uses `ThingOS` CMake system name |
| `patches/rust/llvm-project/0001-llvm-classify-thingos-as-unix.patch` | `vendor/rust/src/llvm-project` | Classify ThingOS as Unix in LLVM CMake |

### Legacy flat patches (documentation only — NOT applied automatically)

The numbered `patches/rust/*.patch` files (`00-core-prelude.patch` through
`95-net.patch`) are historical snapshots of what was once applied manually.
They are **not** replayed by `just rust-apply-patches`.  The PAL
implementation they document (`library/std/src/sys/pal/thingos/`) is committed
directly to the `dancxjo/rust-thingos` fork.

---

## Current artifact layout

After a successful `cargo xtask rustc-thingos`, the bootstrap
writes artifacts to the repository-root `build/` tree:

```text
build/x86_64-unknown-linux-gnu/stage1-rustc/x86_64-unknown-linux-gnu/release/rustc-main
build/x86_64-unknown-linux-gnu/stage1-rustc/x86_64-unknown-linux-gnu/release/deps/librustc_driver-*.rlib
build/x86_64-unknown-linux-gnu/stage1-std/x86_64-unknown-thingos/release/deps/*.rlib
build/x86_64-unknown-linux-gnu/stage1/lib/rustlib/x86_64-unknown-linux-gnu/...
```

The `xtask` wrapper then assembles a developer-friendly cache at:

```text
target/rustc-thingos/rustc              ← stage-1 rustc binary (Linux ELF)
target/rustc-thingos/rustc-wrapper      ← wrapper script (sets sysroot + LD_LIBRARY_PATH)
target/rustc-thingos/lib/rustlib/       ← host + ThingOS target sysroot
target/rustc-thingos/.cache-key         ← invalidation hash
```

The cache is keyed on `targets/x86_64-unknown-thingos.json`,
`rust-toolchain.toml`, and the git HEAD of `vendor/rust`.

---

## Important limitation: Linux-hosted cross-compiler only

The current bootstrap config:

```toml
[build]
host  = ["x86_64-unknown-linux-gnu"]
target = ["x86_64-unknown-linux-gnu", "x86_64-unknown-thingos"]
```

produces a **Linux-hosted cross-compiler** (`ELF 64-bit LSB pie executable,
dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2`).

Shipping this binary into the ThingOS ISO would put an unusable Linux ELF at
`/bin/rustc`, so ISO staging is intentionally disabled pending a
ThingOS-native compiler.

---

## Known remaining issues

### 1. `unexpected_cfgs` warnings

The std build emits `unexpected_cfgs` warnings for `#[cfg(target_os =
"thingos")]` in the PAL modules.  The warnings do not block the build but
indicate `thingos` is not yet declared in `library/std/build.rs` as a known
`target_os` value.

**Fix needed**: a new `vendor-rust/` patch that adds a
`cargo::rustc-check-cfg=cfg(target_os, values("thingos"))` directive to
`library/std/build.rs` (or equivalent in the compiler's check-cfg list).

### 2. ThingOS-hosted compiler

To ship `rustc` inside the ThingOS image, the bootstrap `host` must be changed
to `x86_64-unknown-thingos`:

```toml
[build]
host = ["x86_64-unknown-thingos"]
target = ["x86_64-unknown-thingos"]
```

This requires the full ThingOS userspace runtime (process spawn, VFS, signal
handling) to be stable enough to host a compiler process.

### 3. Canonical rustlib sysroot

ThingOS target `.rlib` files currently come from the cargo output directory
`stage1-std/x86_64-unknown-thingos/release/deps/` rather than the canonical
`stage1/lib/rustlib/x86_64-unknown-thingos/lib/` path.  The xtask wrapper
reconstructs a synthetic sysroot from the deps directory.  A cleaner approach
is to promote the ThingOS sysroot to the canonical path via a bootstrap patch.

### 4. ISO staging (blocked on items 2 and 3)

`stage_rustc_for_iso` in `xtask/src/rustc_thingos.rs` is a no-op today.
Once a ThingOS-native compiler exists it should:

1. Copy `rustc` to `<iso_root>/bin/rustc`.
2. Copy only `rustlib/x86_64-unknown-thingos/lib/` to
   `<iso_root>/usr/lib/rustlib/x86_64-unknown-thingos/lib/` (not the full
   linux-gnu sysroot).

---

## Checkout / vendor/rust note

Some checkouts have `.gitmodules` and `.git/modules/vendor/rust` present
without a `vendor/rust` gitlink tracked in the main worktree index.  In that
state `just fetch-rust` falls back to a plain `git clone` of the fork rather
than a submodule update; both modes are supported.

---

*Last updated: April 12, 2026*
*Status: Linux-hosted cross-compiler bootstrap succeeds; LLVM blockers resolved;
ThingOS-native compiler and ISO staging pending.*
