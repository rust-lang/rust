// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

pub trait Tr {
    fn f();
    const C: u8;
    type T;
}
pub struct S {
    pub a: u8
}
struct Ts(pub u8);

pub impl Tr for S {  //~ ERROR unnecessary visibility qualifier
    pub fn f() {} //~ ERROR unnecessary visibility qualifier
    pub const C: u8 = 0; //~ ERROR unnecessary visibility qualifier
    pub type T = u8; //~ ERROR unnecessary visibility qualifier
}
pub impl S { //~ ERROR unnecessary visibility qualifier
    pub fn f() {}
    pub const C: u8 = 0;
    // pub type T = u8;
}
pub extern "C" { //~ ERROR unnecessary visibility qualifier
    pub fn f();
    pub static St: u8;
}

const MAIN: u8 = {
    pub trait Tr {
        fn f();
        const C: u8;
        type T;
    }
    pub struct S {
        pub a: u8
    }
    struct Ts(pub u8);

    pub impl Tr for S {  //~ ERROR unnecessary visibility qualifier
        pub fn f() {} //~ ERROR unnecessary visibility qualifier
        pub const C: u8 = 0; //~ ERROR unnecessary visibility qualifier
        pub type T = u8; //~ ERROR unnecessary visibility qualifier
    }
    pub impl S { //~ ERROR unnecessary visibility qualifier
        pub fn f() {}
        pub const C: u8 = 0;
        // pub type T = u8;
    }
    pub extern "C" { //~ ERROR unnecessary visibility qualifier
        pub fn f();
        pub static St: u8;
    }

    0
};

fn main() {
    pub trait Tr {
        fn f();
        const C: u8;
        type T;
    }
    pub struct S {
        pub a: u8
    }
    struct Ts(pub u8);

    pub impl Tr for S {  //~ ERROR unnecessary visibility qualifier
        pub fn f() {} //~ ERROR unnecessary visibility qualifier
        pub const C: u8 = 0; //~ ERROR unnecessary visibility qualifier
        pub type T = u8; //~ ERROR unnecessary visibility qualifier
    }
    pub impl S { //~ ERROR unnecessary visibility qualifier
        pub fn f() {}
        pub const C: u8 = 0;
        // pub type T = u8;
    }
    pub extern "C" { //~ ERROR unnecessary visibility qualifier
        pub fn f();
        pub static St: u8;
    }
}
