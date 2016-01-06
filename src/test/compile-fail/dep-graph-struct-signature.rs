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

// compile-flags: -Z incr-comp

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

    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    trait Bar {
        fn do_something(x: WillChange);
    }

    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    fn some_fn(x: WillChange) { }

    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    fn new_foo(x: u32, y: u32) -> WillChange {
        WillChange { x: x, y: y }
    }

    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    impl WillChange {
        fn new(x: u32, y: u32) -> WillChange { loop { } }
    }

    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    impl WillChange {
        fn method(&self, x: u32) { }
    }

    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    struct WillChanges {
        x: WillChange,
        y: WillChange
    }

    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    fn indirect(x: WillChanges) { }
}

// these are invalid dependencies, though sometimes we create edges
// anyway.
mod invalid_signatures {
    use WontChange;

    // FIXME due to the variance pass having overly conservative edges,
    // we incorrectly think changes are needed here
    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    trait A {
        fn do_something_else_twice(x: WontChange);
    }

    // FIXME due to the variance pass having overly conservative edges,
    // we incorrectly think changes are needed here
    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR OK
    fn b(x: WontChange) { }

    #[rustc_then_this_would_need(ItemSignature)] //~ ERROR no path from `WillChange`
    #[rustc_then_this_would_need(CollectItem)] //~ ERROR no path from `WillChange`
    fn c(x: u32) { }
}

