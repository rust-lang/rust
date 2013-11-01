// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name="external_linkage", vers="0.0")];

#[cfg(target_os="macos")]
pub mod linkhack {
    // Lovely hack to get Linuxy shlib behaviour on OSX
    // Will fail with SIGTRAP at runtime if any symbol can't be resolved.
    #[link_args="-Wl,-flat_namespace -Wl,-undefined,suppress"]
    extern {
    }
}

extern "C" {
    fn foreign();
}

extern {
    fn visible();
    static x: int;
}

#[fixed_stack_segment]
pub unsafe fn doer() -> int {
    foreign();
    visible();
    x
}
