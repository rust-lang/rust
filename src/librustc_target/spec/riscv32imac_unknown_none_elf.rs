// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkerFlavor, PanicStrategy, Target, TargetOptions, TargetResult};
use spec::abi::{Abi};

pub fn target() -> TargetResult {
    Ok(Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".to_string(),
        llvm_target: "riscv32".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "none".to_string(),
        target_env: "".to_string(),
        target_vendor: "unknown".to_string(),
        arch: "riscv32".to_string(),
        linker_flavor: LinkerFlavor::Ld,

        options: TargetOptions {
            linker: Some("riscv32-unknown-elf-ld".to_string()),
            cpu: "generic-rv32".to_string(),
            max_atomic_width: Some(32),
            atomic_cas: false, // incomplete +a extension
            features: "+m,+a".to_string(), // disable +c extension
            executables: true,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: "static".to_string(),
            emit_debug_gdb_scripts: false,
            abi_blacklist: vec![
                Abi::Cdecl,
                Abi::Stdcall,
                Abi::Fastcall,
                Abi::Vectorcall,
                Abi::Thiscall,
                Abi::Aapcs,
                Abi::Win64,
                Abi::SysV64,
                Abi::PtxKernel,
                Abi::Msp430Interrupt,
                Abi::X86Interrupt,
            ],
            .. Default::default()
        },
    })
}
