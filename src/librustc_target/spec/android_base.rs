// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut base = super::linux_base::opts();
    // Many of the symbols defined in compiler-rt are also defined in libgcc.
    // Android's linker doesn't like that by default.
    base.pre_link_args
        .get_mut(&LinkerFlavor::Gcc).unwrap().push("-Wl,--allow-multiple-definition".to_string());
    base.is_like_android = true;
    base.position_independent_executables = true;
    base.has_elf_tls = false;
    base.requires_uwtable = true;
    base
}
