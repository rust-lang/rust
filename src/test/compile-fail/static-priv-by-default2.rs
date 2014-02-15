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

fn foo<T>(_: T) {}

fn test1() {
    use child::childs_child::private;
    //~^ ERROR: static `private` is private
    use child::childs_child::public;

    foo(private);
}

fn test2() {
    use static_priv_by_default::private;
    //~^ ERROR: static `private` is private
    use static_priv_by_default::public;

    foo(private);
}

fn main() {}
