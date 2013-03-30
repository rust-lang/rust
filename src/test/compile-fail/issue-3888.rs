// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// n.b. This should be a run-pass test, but for now I'm testing
// that we don't see an "unknown scope" error.
fn vec_peek<'r, T>(v: &'r [T]) -> Option< (&'r T, &'r [T]) > {
    if v.len() == 0 {
        None
    } else {
        let vec_len = v.len();
        let head = &v[0];
        // note: this *shouldn't* be an illegal borrow! See #3888
        let tail = v.slice(1, vec_len); //~ ERROR illegal borrow: borrowed value does not live long enough
        Some( (head, tail) )
    }
}


fn test_peek_empty_stack() {
    let v : &[int] = &[];
    assert!((None == vec_peek(v)));
}

fn test_peek_empty_unique() {
    let v : ~[int] = ~[];
    assert!((None == vec_peek(v)));
}

fn test_peek_empty_managed() {
    let v : @[int] = @[];
    assert!((None == vec_peek(v)));
}


fn main() {}
