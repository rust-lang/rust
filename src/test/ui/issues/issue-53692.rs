// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
fn main() {
        let items = vec![1, 2, 3];
        let ref_items: &[i32] = &items;
        let items_clone: Vec<i32> = ref_items.clone();

        // in that case no suggestion will be triggered
        let items_clone_2:Vec<i32> = items.clone();

        let s = "hi";
        let string: String = s.clone();

        // in that case no suggestion will be triggered
        let s2 = "hi";
        let string_2: String = s2.to_string();
}
