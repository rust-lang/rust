// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// See x86_64_unknown_linux_musl for explanation of arguments

use target::Target;

pub fn target() -> Target {
    let mut base = super::linux_base::opts();
    base.cpu = "pentium4".to_string();
    base.pre_link_args.push("-m32".to_string());
    base.pre_link_args.push("-Wl,-melf_i386".to_string());

    base.pre_link_args.push("-nostdlib".to_string());
    base.pre_link_args.push("-static".to_string());
    base.pre_link_args.push("-Wl,--eh-frame-hdr".to_string());

    base.pre_link_args.push("-Wl,-(".to_string());
    base.post_link_args.push("-Wl,-)".to_string());

    base.pre_link_objects_exe.push("crt1.o".to_string());
    base.pre_link_objects_exe.push("crti.o".to_string());
    base.post_link_objects.push("crtn.o".to_string());

    base.dynamic_linking = false;
    base.has_rpath = false;
    base.position_independent_executables = false;

    Target {
        llvm_target: "i686-unknown-linux-musl".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128".to_string(),
        arch: "x86".to_string(),
        target_os: "linux".to_string(),
        target_env: "musl".to_string(),
        target_vendor: "unknown".to_string(),
        options: base,
    }
}
