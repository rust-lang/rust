// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:static_priv_by_default.rs

extern crate static_priv_by_default;

mod child {
    pub mod childs_child {
        static private: int = 0;
        pub static public: int = 0;
    }
}

fn foo(_: int) {}

fn full_ref() {
    foo(static_priv_by_default::private); //~ ERROR: static `private` is private
    foo(static_priv_by_default::public);
    foo(child::childs_child::private); //~ ERROR: static `private` is private
    foo(child::childs_child::public);
}

fn medium_ref() {
    use child::childs_child;
    foo(childs_child::private); //~ ERROR: static `private` is private
    foo(childs_child::public);
}

fn main() {}
