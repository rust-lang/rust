// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkerFlavor, Target, TargetResult, PanicStrategy, LldFlavor};

pub fn target() -> TargetResult {
    let mut base = super::windows_msvc_base::opts();
    base.max_atomic_width = Some(64);
    base.has_elf_tls = true;

    // FIXME: this shouldn't be panic=abort, it should be panic=unwind
    base.panic_strategy = PanicStrategy::Abort;
    base.linker = Some("rust-lld".to_owned());

    Ok(Target {
        llvm_target: "aarch64-pc-windows-msvc".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        target_os: "windows".to_string(),
        target_env: "msvc".to_string(),
        target_vendor: "pc".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Link),
        options: base,
    })
}
