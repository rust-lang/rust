// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const QUERY = 'String';

const EXPECTED = {
    'others': [
        { 'path': 'std::string', 'name': 'String' },
        { 'path': 'std::ffi', 'name': 'CString' },
        { 'path': 'std::ffi', 'name': 'OsString' },
    ],
    'in_args': [
        { 'path': 'std::str', 'name': 'eq' },
    ],
    'returned': [
        { 'path': 'std::string::String', 'name': 'add' },
    ],
};
