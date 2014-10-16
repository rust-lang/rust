// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo(c: Vec<int> ) {
    let a: int = 5;
    let mut b: Vec<int> = Vec::new();


    match none::<int> {
        some::<int>(_) => {
            for _i in c.iter() {
                println!("{}", a);
                let a = 17i;
                b.push(a);
            }
        }
        _ => { }
    }
}

enum t<T> { none, some(T), }

pub fn main() { let x = 10i; let x = x + 20; assert!((x == 30)); foo(Vec::new()); }
