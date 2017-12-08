// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// These 4 `thumbv*` targets cover the ARM Cortex-M family of processors which are widely used in
// microcontrollers. Namely, all these processors:
//
// - Cortex-M0
// - Cortex-M0+
// - Cortex-M1
// - Cortex-M3
// - Cortex-M4(F)
// - Cortex-M7(F)
//
// We have opted for 4 targets instead of one target per processor (e.g. `cortex-m0`, `cortex-m3`,
// etc) because the differences between some processors like the cortex-m0 and cortex-m1 are almost
// non-existent from the POV of codegen so it doesn't make sense to have separate targets for them.
// And if differences exist between two processors under the same target, rustc flags can be used to
// optimize for one processor or the other.
//
// Also, we have not chosen a single target (`arm-none-eabi`) like GCC does because this makes
// difficult to integrate Rust code and C code. Targeting the Cortex-M4 requires different gcc flags
// than the ones you would use for the Cortex-M0 and with a single target it'd be impossible to
// differentiate one processor from the other.
//
// About arm vs thumb in the name. The Cortex-M devices only support the Thumb instruction set,
// which is more compact (higher code density), and not the ARM instruction set. That's why LLVM
// triples use thumb instead of arm. We follow suit because having thumb in the name let us
// differentiate these targets from our other `arm(v7)-*-*-gnueabi(hf)` targets in the context of
// build scripts / gcc flags.

use std::default::Default;
use spec::{PanicStrategy, TargetOptions};

pub fn opts() -> TargetOptions {
    // See rust-lang/rfcs#1645 for a discussion about these defaults
    TargetOptions {
        executables: true,
        // In 99%+ of cases, we want to use the `arm-none-eabi-gcc` compiler (there aren't many
        // options around)
        linker: Some("arm-none-eabi-gcc".to_string()),
        // Because these devices have very little resources having an unwinder is too onerous so we
        // default to "abort" because the "unwind" strategy is very rare.
        panic_strategy: PanicStrategy::Abort,
        // Similarly, one almost always never wants to use relocatable code because of the extra
        // costs it involves.
        relocation_model: "static".to_string(),
        abi_blacklist: super::arm_base::abi_blacklist(),
        // When this section is added a volatile load to its start address is also generated. This
        // volatile load is a footgun as it can end up loading an invalid memory address, depending
        // on how the user set up their linker scripts. This section adds pretty printer for stuff
        // like std::Vec, which is not that used in no-std context, so it's best to left it out
        // until we figure a way to add the pretty printers without requiring a volatile load cf.
        // rust-lang/rust#44993.
        emit_debug_gdb_scripts: false,
        .. Default::default()
    }
}
