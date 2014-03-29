// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

// Regression test for issue #6633
mod issue6633 {
    use self::foo::name::name; //~ ERROR: unused import
    use self::foo::name;

    pub mod foo {
        pub mod name {
            pub type a = int;
            pub mod name {
                pub type a = f64;
            }
        }
    }

    fn bar() -> name::a { 1 }
}

// Regression test for issue #6935
mod issue6935 {
    use self::a::foo::a::foo;
    use self::a::foo; //~ ERROR: unused import

    pub mod a {
        pub mod foo {
            pub mod a {
                pub fn foo() {}
            }
        }
    }

    fn bar() { foo(); }
}

fn main(){}
