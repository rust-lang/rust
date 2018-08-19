// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const QUERY = 'String::from_ut';

const EXPECTED = {
    'others': [
        { 'path': 'std::string::String', 'name': 'from_utf8' },
        { 'path': 'std::string::String', 'name': 'from_utf8' },
        { 'path': 'std::string::String', 'name': 'from_utf8_lossy' },
        { 'path': 'std::string::String', 'name': 'from_utf16_lossy' },
        { 'path': 'std::string::String', 'name': 'from_utf8_unchecked' },
    ],
};
