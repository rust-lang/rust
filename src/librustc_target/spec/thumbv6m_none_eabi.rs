// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Targets the Cortex-M0, Cortex-M0+ and Cortex-M1 processors (ARMv6-M architecture)

use spec::{LinkerFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "thumbv6m-none-eabi".to_string(),
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
            // The ARMv6-M architecture doesn't support unaligned loads/stores so we disable them
            // with +strict-align.
            features: "+strict-align".to_string(),
            // There are no atomic CAS instructions available in the instruction set of the ARMv6-M
            // architecture
            atomic_cas: false,
            .. super::thumb_base::opts()
        }
    })
}
