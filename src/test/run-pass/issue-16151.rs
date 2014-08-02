// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

static mut DROP_COUNT: uint = 0;

struct Fragment;

impl Drop for Fragment {
    fn drop(&mut self) {
        unsafe {
            DROP_COUNT += 1;
        }
    }
}

fn main() {
    {
        let mut fragments = vec![Fragment, Fragment, Fragment];
        let _new_fragments: Vec<Fragment> = mem::replace(&mut fragments, vec![])
            .move_iter()
            .skip_while(|_fragment| {
                true
            }).collect();
    }
    unsafe {
        assert_eq!(DROP_COUNT, 3);
    }
}

