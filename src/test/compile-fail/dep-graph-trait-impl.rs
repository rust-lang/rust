// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that when a trait impl changes, fns whose body uses that trait
// must also be recompiled.

// compile-flags: -Z incr-comp

#![feature(rustc_attrs)]
#![allow(warnings)]

fn main() { }

pub trait Foo: Sized {
    fn method(self) { }
}

mod x {
    use Foo;

    #[rustc_if_this_changed]
    impl Foo for char { }

    impl Foo for u32 { }
}

mod y {
    use Foo;

    #[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR OK
    #[rustc_then_this_would_need(TransCrateItem)] //~ ERROR OK
    pub fn with_char() {
        char::method('a');
    }

    // FIXME(#30741) tcx fulfillment cache not tracked
    #[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR no path
    #[rustc_then_this_would_need(TransCrateItem)] //~ ERROR no path
    pub fn take_foo_with_char() {
        take_foo::<char>('a');
    }

    #[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR OK
    #[rustc_then_this_would_need(TransCrateItem)] //~ ERROR OK
    pub fn with_u32() {
        u32::method(22);
    }

    // FIXME(#30741) tcx fulfillment cache not tracked
    #[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR no path
    #[rustc_then_this_would_need(TransCrateItem)] //~ ERROR no path
    pub fn take_foo_with_u32() {
        take_foo::<u32>(22);
    }

    pub fn take_foo<T:Foo>(t: T) { }
}

mod z {
    use y;

    // These are expected to yield errors, because changes to `x`
    // affect the BODY of `y`, but not its signature.
    #[rustc_then_this_would_need(TypeckItemBody)] //~ ERROR no path
    #[rustc_then_this_would_need(TransCrateItem)] //~ ERROR no path
    pub fn z() {
        y::with_char();
        y::with_u32();
    }
}
