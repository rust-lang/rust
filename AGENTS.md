# Thing-OS Agent Notes

This file is a quick map of the repository so agents (and humans) can orient fast.

## What this repo is
- Thing-OS is a Rust workspace that builds a VFS-first OS kernel plus userland apps.
- The old kernel graph/ThingId model is legacy. Do not introduce new boot, display, or UI dependencies on `stem::thing`, `ThingId`, `UI_CROWN`, or graph discovery for core system bring-up.
- Build/test automation lives in `xtask` and is surfaced via `just`.

## Common commands
- Build ISO: `just iso` (override arch with `KARCH=aarch64`, `riscv64`, `loongarch64`)
- Run QEMU: `just run`
- Run BDD tests: `just behave` (see `tools/bdd`)
- Clean: `just clean`
- Audit platform boundary: `python3 scripts/audit_platform_boundary.py`
- Rust source checkout: `just fetch-rust`

## Top-level layout (what's what)
- `abi/`: shared ABI types and syscalls between kernel/userspace.
- `bran/`: core kernel runtime (boot/runtime abstraction).
- `kernel/`: kernel crate and core kernel logic.
- `drivers/`: hardware driver crates.
- `userspace/`: user programs and demos (each subdir is a crate).
- `bloom/`, `blossom/`, `display/`: graphics/compositor-related crates.
- `stem/`, `stem-macros/`: internal libs and proc-macros.
- `targets/`: custom JSON target specs for bare metal builds.
- `tools/`: auxiliary tooling (BDD, pciids, etc).
- `xtask/`: build orchestration used by `just`.
- `docs/`: documentation and test reports (`docs/behavior/` is generated).
- `vendor/`: vendored dependencies (Limine, OVMF).

## Where to start when changing behavior
- Kernel interfaces: `abi/` and `kernel/`
- Syscall surface: `abi/src/syscall.rs`
- User apps: `userspace/`
- Build/config: `justfile`, `xtask/`, `targets/`

## UI / Display Contract

- Display discovery and presentation are filesystem-driven.
- The kernel exposes the boot framebuffer at `/dev/fb0`; display drivers and the compositor must bind through files and file descriptors, not graph nodes.
- Bloom should boot and paint with only VFS/device state available. Do not require `UI_CROWN`, graph watches, or `ThingId` lookups to reach first paint.
- Runtime UI coordination should happen through mounted services and session/runtime files such as `/services`, `/run`, and `/session`.
- Desktop background configuration lives at `/session/desktop/{wallpaper,mode,background_color}` and Bloom is expected to watch and react to those files.

## Platform Layer Contract ("stem is our std")

**Thing-OS does not use Rust's `std`**. Instead:

- Kernel and userspace use `core` + `alloc` + `stem`
- Platform capabilities are explicit in `stem::pal` (Platform Abstraction Layer)
- Build tools (`xtask`, `tools/*`) can use `std` (they run at compile-time only)

**Key rules:**
- All kernel/userspace crates MUST have `#![no_std]`
- Platform primitives go in `stem::pal` (log, clock, abort, alloc)
- Run `python3 scripts/audit_platform_boundary.py` to verify compliance

**See `docs/platform.md` for the complete platform layer contract.**

## Rust source of truth

Thing-OS uses a fork of the Rust compiler and standard library to support its custom target triple and VFS-first architecture.

- **Fork Repository**: [dancxjo/rust-thingos](https://github.com/dancxjo/rust-thingos)
- **Local Path**: `vendor/rust/` (populated via `just fetch-rust`)
- **Modifications**: All changes to `core`, `alloc`, `std`, or the compiler must be committed directly to the `rust-thingos` fork. This repository does not use local `.patch` files.
- **Submodules**: Manual changes to submodules (like LLVM) are documented in `vendor/rust/submodule_patches.md`.

Workflow:
1. `just fetch-rust` (initializes/syncs the `vendor/rust` submodule and pins it to `thingos-patched`)
2. Edit `vendor/rust/...`
3. Commit and push changes to the `rust-thingos` fork repository.
4. Run `just rust-reset` to hard-reset `vendor/rust` to `origin/thingos-patched`.

Important implications:
- Fresh checkouts do not have `vendor/rust/`.
- `git status` in the main repo does not track changes inside `vendor/rust/`.
- The `xtask` build system hashes the git revision of `vendor/rust/` to detect when the compiler needs to be rebuilt.

## Architecture Guardrails

Four non-negotiable design rules govern all kernel and userspace changes:

1. **Scheduler-first** — every unit of execution is a kernel-scheduled task.
2. **Userland drivers** — hardware logic lives in userspace; kernel exposes only `SYS_DEVICE_*` primitives.
3. **VFS-first** — all system resources are reachable through mounted filesystem paths.
4. **Spawn + exec** — new processes use `SYS_SPAWN_PROCESS[_EX]` + `SYS_TASK_EXEC`; there is no `SYS_FORK`.

**See `docs/concepts/janix-guardrails.md` for the full reference and PR review checklist.**

## Notes
- Workspace members are listed in `Cargo.toml`.
- `target/` is build output and can be ignored in reviews.
- Reminder: use the `apply_patch` tool directly for file edits (avoid running it via exec). 
- Reminder: hashing helpers must use the correct byte width for each integer type (u32/i32 = 4 bytes).
- The `stem` build script expects `assets/pci/pci.ids`; ensure it exists (or skip builds that trigger `stem`'s build.rs) when running `cargo test`.
- Reminder: avoid clearing all UI caches on watch events; prefer targeted invalidation once event payloads provide node or bytespace IDs.
