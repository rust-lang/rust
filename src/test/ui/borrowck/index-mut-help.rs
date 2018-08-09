// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// When mutably indexing a type that implements `Index` but not `IndexMut`, a
// special 'help' message is added to the output.


fn main() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    map.insert("peter", "23".to_string());

    map["peter"].clear();           //~ ERROR
    map["peter"] = "0".to_string(); //~ ERROR
    let _ = &mut map["peter"];      //~ ERROR
}
