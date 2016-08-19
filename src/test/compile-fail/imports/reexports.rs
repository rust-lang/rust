// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(item_like_imports)]

mod a {
    fn foo() {}
    mod foo {}

    mod a {
        pub use super::foo; //~ ERROR cannot be reexported
    }
}

mod b {
    pub fn foo() {}
    mod foo { pub struct S; }

    pub mod a {
        pub use super::foo; // This is OK since the value `foo` is visible enough.
        fn f(_: foo::S) {} // `foo` is imported in the type namespace (but not `pub` reexported).
    }
}

mod c {
    // Test that `foo` is not reexported.
    use b::a::foo::S; //~ ERROR `foo`
}

fn main() {}
