// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]

/// Creates a `std::vec::Vec` containing the arguments.
macro_rules! vec(
    ($($e:expr),*) => ({
        // leading _ to allow empty construction without a warning.
        let mut _temp = ::vec::Vec::new();
        $(_temp.push($e);)*
        _temp
    });
    ($($e:expr),+,) => (vec!($($e),+))
)
