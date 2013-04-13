// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for `u8`

pub use self::inst::is_ascii;

mod inst {
    pub type T = u8;
    #[allow(non_camel_case_types)]
    pub type T_SIGNED = i8;
    pub static bits: uint = 8;

    // Type-specific functions here. These must be reexported by the
    // parent module so that they appear in core::u8 and not core::u8::u8;

    pub fn is_ascii(x: T) -> bool { return 0 as T == x & 128 as T; }
}
