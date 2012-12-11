// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This tests that exports can have visible dependencies on things
// that are not exported, allowing for a sort of poor-man's ADT

mod foo {
    #[legacy_exports];
    export f;
    export g;

    // not exported
    enum t { t1, t2, }

    impl t : cmp::Eq {
        pure fn eq(&self, other: &t) -> bool {
            ((*self) as uint) == ((*other) as uint)
        }
        pure fn ne(&self, other: &t) -> bool { !(*self).eq(other) }
    }

    fn f() -> t { return t1; }

    fn g(v: t) { assert (v == t1); }
}

fn main() { foo::g(foo::f()); }
