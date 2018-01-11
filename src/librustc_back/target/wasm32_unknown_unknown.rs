// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The wasm32-unknown-unknown target is currently a highly experimental version
// of a wasm-based target which does *not* use the Emscripten toolchain. Instead
// this is a pretty flavorful (aka hacked up) target right now. The definition
// and semantics of this target are likely to change and so this shouldn't be
// relied on just yet.
//
// In general everyone is currently waiting on a linker for wasm code. In the
// meantime we have no means of actually making use of the traditional separate
// compilation model. At a high level this means that assembling Rust programs
// into a WebAssembly program looks like:
//
//  1. All intermediate artifacts are LLVM bytecode. We'll be using LLVM as
//     a linker later on.
//  2. For the final artifact we emit one giant assembly file (WebAssembly
//     doesn't have an object file format). To do this we force LTO to be turned
//     on (`requires_lto` below) to ensure all Rust code is in one module. Any
//     "linked" C library is basically just ignored.
//  3. Using LLVM we emit a `foo.s` file (assembly) with some... what I can only
//     describe as arcane syntax. From there we need to actually change this
//     into a wasm module. For this step we use the `binaryen` project. This
//     project is mostly intended as a WebAssembly code generator, but for now
//     we're just using its LLVM-assembly-to-wasm-module conversion utilities.
//
// And voila, out comes a web assembly module! There's some various tweaks here
// and there, but that's the high level at least. Note that this will be
// rethought from the ground up once a linker (lld) is available, so this is all
// temporary and should improve in the future.

use LinkerFlavor;
use super::{Target, TargetOptions, PanicStrategy};

pub fn target() -> Result<Target, String> {
    let opts = TargetOptions {
        linker: "not-used".to_string(),

        // we allow dynamic linking, but only cdylibs. Basically we allow a
        // final library artifact that exports some symbols (a wasm module) but
        // we don't allow intermediate `dylib` crate types
        dynamic_linking: true,
        only_cdylib: true,

        // This means we'll just embed a `start` function in the wasm module
        executables: true,

        // relatively self-explanatory!
        exe_suffix: ".wasm".to_string(),
        dll_prefix: "".to_string(),
        dll_suffix: ".wasm".to_string(),
        linker_is_gnu: false,

        // We're storing bitcode for now in all the rlibs
        obj_is_bitcode: true,

        // A bit of a lie, but "eh"
        max_atomic_width: Some(32),

        // Unwinding doesn't work right now, so the whole target unconditionally
        // defaults to panic=abort. Note that this is guaranteed to change in
        // the future once unwinding is implemented. Don't rely on this.
        panic_strategy: PanicStrategy::Abort,

        // There's no linker yet so we're forced to use LLVM as a linker. This
        // means that we must always enable LTO for final artifacts.
        requires_lto: true,

        // Wasm doesn't have atomics yet, so tell LLVM that we're in a single
        // threaded model which will legalize atomics to normal operations.
        singlethread: true,

        // Because we're always enabling LTO we can't enable builtin lowering as
        // otherwise we'll lower the definition of the `memcpy` function to
        // memcpy itself. Note that this is specifically because we're
        // performing LTO with compiler-builtins.
        no_builtins: true,

        .. Default::default()
    };
    Ok(Target {
        llvm_target: "wasm32-unknown-unknown".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        // This is basically guaranteed to change in the future, don't rely on
        // this. Use `not(target_os = "emscripten")` for now.
        target_os: "unknown".to_string(),
        target_env: "".to_string(),
        target_vendor: "unknown".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-n32:64-S128".to_string(),
        arch: "wasm32".to_string(),
        // A bit of a lie, but it gets the job done
        linker_flavor: LinkerFlavor::Binaryen,
        options: opts,
    })
}
