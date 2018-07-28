// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkArgs, LinkerFlavor, TargetOptions};
use std::default::Default;

pub fn opts() -> TargetOptions {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Gcc, vec![
        // We want to be able to strip as much executable code as possible
        // from the linker command line, and this flag indicates to the
        // linker that it can avoid linking in dynamic libraries that don't
        // actually satisfy any symbols up to that point (as with many other
        // resolutions the linker does). This option only applies to all
        // following libraries so we're sure to pass it as one of the first
        // arguments.
        // FIXME: figure out whether these linker args are desirable
        //"-Wl,--as-needed".to_string(),

        // Always enable NX protection when it is available
        //"-Wl,-z,noexecstack".to_string(),
    ]);

    TargetOptions {
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: false,
        pre_link_args: args,
        position_independent_executables: true,
        has_elf_tls: true,
        .. Default::default()
    }
}
