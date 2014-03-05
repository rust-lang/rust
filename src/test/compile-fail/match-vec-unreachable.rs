// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let x: ~[(int, int)] = ~[];
    let x: &[(int, int)] = x;
    match x {
        [a, (2, 3), _] => (),
        [(1, 2), (2, 3), b] => (), //~ ERROR unreachable pattern
        _ => ()
    }

    let x: ~[~str] = ~[~"foo", ~"bar", ~"baz"];
    let x: &[~str] = x;
    match x {
        [a, _, _, ..] => { println!("{}", a); }
        [~"foo", ~"bar", ~"baz", ~"foo", ~"bar"] => { } //~ ERROR unreachable pattern
        _ => { }
    }

    let x: ~[char] = ~['a', 'b', 'c'];
    let x: &[char] = x;
    match x {
        ['a', 'b', 'c', .._tail] => {}
        ['a', 'b', 'c'] => {} //~ ERROR unreachable pattern
        _ => {}
    }
}
