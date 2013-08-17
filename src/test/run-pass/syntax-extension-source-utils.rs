// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is brittle!
// xfail-pretty - the pretty tests lose path information, breaking include!

pub mod m1 {
    pub mod m2 {
        pub fn where_am_i() -> ~str { (module_path!()).to_owned() }
    }
}

macro_rules! indirect_line( () => ( line!() ) )

pub fn main() {
    assert_eq!(line!(), 23);
    //assert!((col!() == 11));
    assert_eq!(indirect_line!(), 25);
    assert!((file!().to_owned().ends_with("syntax-extension-source-utils.rs")));
    assert_eq!(stringify!((2*3) + 5).to_owned(), ~"( 2 * 3 ) + 5");
    assert!(include!("syntax-extension-source-utils-files/includeme.fragment").to_owned()
           == ~"victory robot 6");

    assert!(
        include_str!("syntax-extension-source-utils-files/includeme.fragment").to_owned()
        .starts_with("/* this is for "));
    assert!(
        include_bin!("syntax-extension-source-utils-files/includeme.fragment")
        [1] == (42 as u8)); // '*'
    // The Windows tests are wrapped in an extra module for some reason
    assert!((m1::m2::where_am_i().ends_with("m1::m2")));
}
