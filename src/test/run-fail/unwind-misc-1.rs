// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_NEWRT=1
// error-pattern:fail

fn main() {
    let count = @mut 0u;
    let mut map = std::hashmap::HashMap::new();
    let mut arr = ~[];
    for i in range(0u, 10u) {
        arr.push(@~"key stuff");
        map.insert(arr.clone(), arr + &[@~"value stuff"]);
        if arr.len() == 5 {
            fail!();
        }
    }
}
