// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use test::Bencher;

// FIXME #13642 (these benchmarks should be in another place)
// Completely miscellaneous language-construct benchmarks.
// Static/dynamic method dispatch

struct Struct {
    field: isize
}

trait Trait {
    fn method(&self) -> isize;
}

impl Trait for Struct {
    fn method(&self) -> isize {
        self.field
    }
}

#[bench]
fn trait_vtable_method_call(b: &mut Bencher) {
    let s = Struct { field: 10 };
    let t = &s as &Trait;
    b.iter(|| {
        t.method()
    });
}

#[bench]
fn trait_static_method_call(b: &mut Bencher) {
    let s = Struct { field: 10 };
    b.iter(|| {
        s.method()
    });
}

// Overhead of various match forms

#[bench]
fn match_option_some(b: &mut Bencher) {
    let x = Some(10);
    b.iter(|| {
        match x {
            Some(y) => y,
            None => 11
        }
    });
}

#[bench]
fn match_vec_pattern(b: &mut Bencher) {
    let x = [1,2,3,4,5,6];
    b.iter(|| {
        match x {
            [1,2,3,..] => 10,
            _ => 11,
        }
    });
}
