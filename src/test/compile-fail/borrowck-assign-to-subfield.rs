// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

fn main() {
    struct A {
        a: int,
        w: B,
        x: @B,
        z: @mut B
    }
    struct B {
        a: int
    }
    let mut p = A {
        a: 1,
        w: B {a: 1},
        x: @B {a: 1},
        z: @mut B {a: 1}
    };

    // even though `x` is not declared as a mutable field,
    // `p` as a whole is mutable, so it can be modified.
    p.a = 2;

    // this is true for an interior field too
    p.w.a = 2;

    // in these cases we pass through a box, so the mut
    // of the box is dominant
    p.x.a = 2;     //~ ERROR cannot assign to immutable field
    p.z.a = 2;
}
