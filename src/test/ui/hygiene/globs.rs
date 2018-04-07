// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(decl_macro)]

mod foo {
    pub fn f() {}
}

mod bar {
    pub fn g() {}
}

macro m($($t:tt)*) {
    $($t)*
    use foo::*;
    f();
    g(); //~ ERROR cannot find function `g` in this scope
}

fn main() {
    m! {
        use bar::*;
        g();
        f(); //~ ERROR cannot find function `f` in this scope
    }
}

n!(f);
macro n($i:ident) {
    mod foo {
        pub fn $i() -> u32 { 0 }
        pub fn f() {}

        mod test {
            use super::*;
            fn g() {
                let _: u32 = $i();
                let _: () = f();
            }
        }

        macro n($j:ident) {
            mod test {
                use super::*;
                fn g() {
                    let _: u32 = $i();
                    let _: () = f();
                    $j();
                }
            }
        }

        n!(f);
        mod test2 {
            super::n! {
                f //~ ERROR cannot find function `f` in this scope
            }
        }
    }
}
