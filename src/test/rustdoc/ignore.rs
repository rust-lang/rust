// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:--test

#[doc(test(ignore))]
pub mod foo {
    /**
     * This doc example should be ignored.
     *
     * ```
     * this is invalid rust, and thus would fail the test.
     * ```
     */
    pub fn bar() { }

    /// Just like this.
    ///
    /// ```
    /// moar invalid code
    /// ```
    pub struct Foo(());

    impl Foo {
        /// And this too.
        ///
        /// ```rust
        /// void* foo = bar();
        /// foo->do_baz();
        /// ```
        pub fn baz(&self) -> i32 {
            unreachable!();
        }
    }
}

pub mod boo {
    /// This one should run though.
    ///
    /// ```
    /// let foo = 0xbadc0de;
    /// ```
    pub fn bar() {}

    /// But this should be ignored.
    ///
    /// ```rust
    /// moar code that wouldn't compile in ages.
    /// ```
    #[doc(test(ignore))]
    pub struct Bar;
}
