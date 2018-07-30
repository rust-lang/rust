// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Platform-specific types, as defined by C.
//!
//! Code that interacts via FFI will almost certainly be using the
//! base types provided by C, which aren't nearly as nicely defined
//! as Rust's primitive types. This module provides types which will
//! match those defined by C, so that code that interacts with C will
//! refer to the correct types.

#![stable(feature = "raw_os", since = "1.1.0")]

#[stable(feature = "raw_os", since = "1.1.0")]
pub use ::core::os::raw::*;

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use any::TypeId;
    use libc;
    use mem;
     macro_rules! ok {
        ($($t:ident)*) => {$(
            assert!(TypeId::of::<libc::$t>() == TypeId::of::<raw::$t>(),
                    "{} is wrong", stringify!($t));
        )*}
    }
     #[test]
    fn same() {
        use os::raw;
        ok!(c_char c_schar c_uchar c_short c_ushort c_int c_uint c_long c_ulong
            c_longlong c_ulonglong c_float c_double);
    }
}
