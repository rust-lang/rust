// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-order

const QUERY = 'is_nan';

const EXPECTED = {
    'others': [
        { 'path': 'std::f32', 'name': 'is_nan' },
        { 'path': 'std::f64', 'name': 'is_nan' },
        { 'path': 'std::option::Option', 'name': 'is_none' },
    ],
};
