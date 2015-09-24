// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A<F: FnOnce()->T,T>(F::Output);
struct B<F: FnOnce()->T,T>(A<F,T>);

// Removing Option causes it to compile.
fn foo<T,F: FnOnce()->T>(f: F) -> Option<B<F,T>> {
    Some(B(A(f())))
}

fn main() {
    let v = (|| foo(||4))();
    match v {
        Some(B(A(4))) => {},
        _ => unreachable!()
    }
}
