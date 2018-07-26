// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Targets the Cortex-M4F and Cortex-M7F processors (ARMv7E-M)
//
// This target assumes that the device does have a FPU (Floating Point Unit) and lowers all (single
// precision) floating point operations to hardware instructions.
//
// Additionally, this target uses the "hard" floating convention (ABI) where floating point values
// are passed to/from subroutines via FPU registers (S0, S1, D0, D1, etc.).
//
// To opt into double precision hardware support, use the `-C target-feature=-fp-only-sp` flag.

use spec::{LinkerFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "thumbv7em-none-eabihf".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: "".to_string(),
        target_vendor: "".to_string(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            // `+vfp4` is the lowest common denominator between the Cortex-M4 (vfp4-16) and the
            // Cortex-M7 (vfp5)
            // `+d16` both the Cortex-M4 and the Cortex-M7 only have 16 double-precision registers
            // available
            // `+fp-only-sp` The Cortex-M4 only supports single precision floating point operations
            // whereas in the Cortex-M7 double precision is optional
            //
            // Reference:
            // ARMv7-M Architecture Reference Manual - A2.5 The optional floating-point extension
            features: "+vfp4,+d16,+fp-only-sp".to_string(),
            max_atomic_width: Some(32),
            .. super::thumb_base::opts()
        }
    })
}
