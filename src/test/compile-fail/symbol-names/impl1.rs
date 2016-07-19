// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(dead_code)]

mod foo {
    pub struct Foo { x: u32 }

    impl Foo {
        #[rustc_symbol_name] //~ ERROR _ZN5impl13foo3Foo3bar
        #[rustc_item_path] //~ ERROR item-path(foo::Foo::bar)
        fn bar() { }
    }
}

mod bar {
    use foo::Foo;

    impl Foo {
        #[rustc_symbol_name] //~ ERROR _ZN5impl13bar33_$LT$impl$u20$impl1..foo..Foo$GT$3baz
        #[rustc_item_path] //~ ERROR item-path(bar::<impl foo::Foo>::baz)
        fn baz() { }
    }
}

fn main() {
}
