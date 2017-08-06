// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test cases where a changing struct appears in the signature of fns
// and methods.

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_variables)]

fn main() { }

#[rustc_if_this_changed]
struct WillChange {
    x: u32,
    y: u32
}

struct WontChange {
    x: u32,
    y: u32
}

// these are valid dependencies
mod signatures {
    use WillChange;

    #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR no path
    #[rustc_then_this_would_need(AssociatedItems)] //~ ERROR no path
    #[rustc_then_this_would_need(TraitDefOfItem)] //~ ERROR no path
    trait Bar {
        #[rustc_then_this_would_need(FnSignature)] //~ ERROR OK
        fn do_something(x: WillChange);
    }

    #[rustc_then_this_would_need(FnSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
    fn some_fn(x: WillChange) { }

    #[rustc_then_this_would_need(FnSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
    fn new_foo(x: u32, y: u32) -> WillChange {
        WillChange { x: x, y: y }
    }

    #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR OK
    impl WillChange {
        #[rustc_then_this_would_need(FnSignature)] //~ ERROR OK
        #[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
        fn new(x: u32, y: u32) -> WillChange { loop { } }
    }

    #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR OK
    impl WillChange {
        #[rustc_then_this_would_need(FnSignature)] //~ ERROR OK
        #[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
        fn method(&self, x: u32) { }
    }

    struct WillChanges {
        #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR OK
        x: WillChange,
        #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR OK
        y: WillChange
    }

    // The fields change, not the type itself.
    #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR no path
    fn indirect(x: WillChanges) { }
}

mod invalid_signatures {
    use WontChange;

    #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR no path
    trait A {
        #[rustc_then_this_would_need(FnSignature)] //~ ERROR no path
        fn do_something_else_twice(x: WontChange);
    }

    #[rustc_then_this_would_need(FnSignature)] //~ ERROR no path
    #[rustc_then_this_would_need(TypeckTables)] //~ ERROR no path
    fn b(x: WontChange) { }

    #[rustc_then_this_would_need(FnSignature)] //~ ERROR no path from `WillChange`
    #[rustc_then_this_would_need(TypeckTables)] //~ ERROR no path from `WillChange`
    fn c(x: u32) { }
}
