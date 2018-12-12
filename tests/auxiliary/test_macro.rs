// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait A {}

macro_rules! __implicit_hasher_test_macro {
    (impl< $($impl_arg:tt),* > for $kind:ty where $($bounds:tt)*) => {
        __implicit_hasher_test_macro!( ($($impl_arg),*) ($kind) ($($bounds)*) );
    };

    (($($impl_arg:tt)*) ($($kind_arg:tt)*) ($($bounds:tt)*)) => {
        impl< $($impl_arg)* > test_macro::A for $($kind_arg)* where $($bounds)* { }
    };
}
