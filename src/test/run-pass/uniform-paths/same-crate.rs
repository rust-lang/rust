// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// edition:2018

pub const A: usize = 0;

pub mod foo {
    pub const B: usize = 1;

    pub mod bar {
        pub const C: usize = 2;

        pub enum E {
            V1(usize),
            V2(String),
        }

        pub fn test() -> String {
            format!("{} {} {}", crate::A, crate::foo::B, C)
        }

        pub fn test_use() -> String {
            use crate::A;
            use crate::foo::B;

            format!("{} {} {}", A, B, C)
        }

        pub fn test_enum() -> String {
            use E::*;
            match E::V1(10) {
                V1(i) => { format!("V1: {}", i) }
                V2(s) => { format!("V2: {}", s) }
            }
        }
    }

    pub fn test() -> String {
        format!("{} {} {}", crate::A, B, bar::C)
    }

    pub fn test_use() -> String {
        use crate::A;
        use bar::C;

        format!("{} {} {}", A, B, C)
    }

    pub fn test_enum() -> String {
        use bar::E::*;
        match bar::E::V1(10) {
            V1(i) => { format!("V1: {}", i) }
            V2(s) => { format!("V2: {}", s) }
        }
    }
}

pub fn test() -> String {
    format!("{} {} {}", A, foo::B, foo::bar::C)
}

pub fn test_use() -> String {
    use foo::B;
    use foo::bar::C;

    format!("{} {} {}", A, B, C)
}

pub fn test_enum() -> String {
    use foo::bar::E::*;
    match foo::bar::E::V1(10) {
        V1(i) => { format!("V1: {}", i) }
        V2(s) => { format!("V2: {}", s) }
    }
}

fn main() {
    let output = [
        test(),
        foo::test(),
        foo::bar::test(),
        test_use(),
        foo::test_use(),
        foo::bar::test_use(),
        test_enum(),
        foo::test_enum(),
        foo::bar::test_enum(),
    ].join("\n");
    assert_eq!(output, "\
0 1 2
0 1 2
0 1 2
0 1 2
0 1 2
0 1 2
V1: 10
V1: 10
V1: 10");
}
