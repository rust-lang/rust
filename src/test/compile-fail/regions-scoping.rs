// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn with<T>(t: T, f: fn(T)) { f(t) }

fn nested(x: &x/int) {  // (1)
    do with(
        fn&(x: &x/int, // Refers to the region `x` at (1)
            y: &y/int, // A fresh region `y` (2)
            z: fn(x: &x/int, // Refers to `x` at (1)
                  y: &y/int, // Refers to `y` at (2)
                  z: &z/int) -> &z/int) // A fresh region `z` (3)
            -> &x/int {

            if false { return z(x, y, x); }

            if false { return z(x, y, y); }
            //~^ ERROR cannot infer an appropriate lifetime

            return z(y, x, x);
            //~^ ERROR mismatched types: expected `&x/int` but found `&y/int`
            //~^^ ERROR mismatched types: expected `&y/int` but found `&x/int`
        }
    ) |foo| {

        let a: &x/int = foo(x, x, |_x, _y, z| z );
        let b: &x/int = foo(x, a, |_x, _y, z| z );
        let c: &x/int = foo(a, a, |_x, _y, z| z );

        let z = 3i;
        let d: &x/int = foo(x, x, |_x, _y, z| z );
        let e: &x/int = foo(x, &z, |_x, _y, z| z );

        // This would result in an error, but it is not reported by typeck
        // anymore but rather borrowck. Therefore, it doesn't end up
        // getting printed out since compilation fails after typeck.
        //
        // let f: &x/int = foo(&z, &z, |_x, _y, z| z ); // ERROR mismatched types: expected `&x/int` but found

        foo(x, &z, |x, _y, _z| x); //~ ERROR mismatched types: expected `&z/int` but found `&x/int`

        // Note: originally I had foo(x, &z, ...) here, but in that
        // case the region inferencer deduced that this was valid if
        // &y==&static, and so inference would succeed but borrow
        // check would fail because the lifetime of &z is not &static.
        foo(x, x, |_x, y, _z| y); //~ ERROR cannot infer an appropriate lifetime
    }
}

fn main() {}