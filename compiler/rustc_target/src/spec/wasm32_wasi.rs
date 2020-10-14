//! The `wasm32-wasi` target is a new and still (as of April 2019) an
//! experimental target. The definition in this file is likely to be tweaked
//! over time and shouldn't be relied on too much.
//!
//! The `wasi` target is a proposal to define a standardized set of syscalls
//! that WebAssembly files can interoperate with. This set of syscalls is
//! intended to empower WebAssembly binaries with native capabilities such as
//! filesystem access, network access, etc.
//!
//! You can see more about the proposal at <https://wasi.dev>.
//!
//! The Rust target definition here is interesting in a few ways. We want to
//! serve two use cases here with this target:
//!
//! * First, we want Rust usage of the target to be as hassle-free as possible,
//!   ideally avoiding the need to configure and install a local wasm32-wasi
//!   toolchain.
//!
//! * Second, one of the primary use cases of LLVM's new wasm backend and the
//!   wasm support in LLD is that any compiled language can interoperate with
//!   any other. To that the `wasm32-wasi` target is the first with a viable C
//!   standard library and sysroot common definition, so we want Rust and C/C++
//!   code to interoperate when compiled to `wasm32-unknown-unknown`.
//!
//! You'll note, however, that the two goals above are somewhat at odds with one
//! another. To attempt to solve both use cases in one go we define a target
//! that (ab)uses the `crt-static` target feature to indicate which one you're
//! in.
//!
//! ## No interop with C required
//!
//! By default the `crt-static` target feature is enabled, and when enabled
//! this means that the bundled version of `libc.a` found in `liblibc.rlib`
//! is used. This isn't intended really for interoperation with a C because it
//! may be the case that Rust's bundled C library is incompatible with a
//! foreign-compiled C library. In this use case, though, we use `rust-lld` and
//! some copied crt startup object files to ensure that you can download the
//! wasi target for Rust and you're off to the races, no further configuration
//! necessary.
//!
//! All in all, by default, no external dependencies are required. You can
//! compile `wasm32-wasi` binaries straight out of the box. You can't, however,
//! reliably interoperate with C code in this mode (yet).
//!
//! ## Interop with C required
//!
//! For the second goal we repurpose the `target-feature` flag, meaning that
//! you'll need to do a few things to have C/Rust code interoperate.
//!
//! 1. All Rust code needs to be compiled with `-C target-feature=-crt-static`,
//!    indicating that the bundled C standard library in the Rust sysroot will
//!    not be used.
//!
//! 2. If you're using rustc to build a linked artifact then you'll need to
//!    specify `-C linker` to a `clang` binary that supports
//!    `wasm32-wasi` and is configured with the `wasm32-wasi` sysroot. This
//!    will cause Rust code to be linked against the libc.a that the specified
//!    `clang` provides.
//!
//! 3. If you're building a staticlib and integrating Rust code elsewhere, then
//!    compiling with `-C target-feature=-crt-static` is all you need to do.
//!
//! You can configure the linker via Cargo using the
//! `CARGO_TARGET_WASM32_WASI_LINKER` env var. Be sure to also set
//! `CC_wasm32-wasi` if any crates in the dependency graph are using the `cc`
//! crate.
//!
//! ## Remember, this is all in flux
//!
//! The wasi target is **very** new in its specification. It's likely going to
//! be a long effort to get it standardized and stable. We'll be following it as
//! best we can with this target. Don't start relying on too much here unless
//! you know what you're getting in to!

use super::wasm32_base;
use super::{crt_objects, LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut options = wasm32_base::options();

    options
        .pre_link_args
        .entry(LinkerFlavor::Gcc)
        .or_insert(Vec::new())
        .push("--target=wasm32-wasi".to_string());

    options.pre_link_objects_fallback = crt_objects::pre_wasi_fallback();
    options.post_link_objects_fallback = crt_objects::post_wasi_fallback();

    // Right now this is a bit of a workaround but we're currently saying that
    // the target by default has a static crt which we're taking as a signal
    // for "use the bundled crt". If that's turned off then the system's crt
    // will be used, but this means that default usage of this target doesn't
    // need an external compiler but it's still interoperable with an external
    // compiler if configured correctly.
    options.crt_static_default = true;
    options.crt_static_respected = true;

    // Allow `+crt-static` to create a "cdylib" output which is just a wasm file
    // without a main function.
    options.crt_static_allows_dylibs = true;

    // WASI's `sys::args::init` function ignores its arguments; instead,
    // `args::args()` makes the WASI API calls itself.
    options.main_needs_argc_argv = false;

    Target {
        llvm_target: "wasm32-wasi".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        target_os: "wasi".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        data_layout: "e-m:e-p:32:32-i64:64-n32:64-S128".to_string(),
        arch: "wasm32".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Wasm),
        options,
    }
}
