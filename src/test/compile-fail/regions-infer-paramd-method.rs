// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test
// ignored due to problems with by value self.

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Here: foo is parameterized because it contains a method that
// refers to self.

trait foo<'a> {
    fn self_int(self) -> &'a int;

    fn any_int(self) -> &int;
}

struct with_foo<'a> {
    f: @foo<'a>
}

trait set_foo_foo {
    fn set_foo(&mut self, f: @foo);
}

impl<'a> set_foo_foo for with_foo<'a> {
    fn set_foo(&mut self, f: @foo) {
        self.f = f; //~ ERROR mismatched types: expected `@foo/&self` but found `@foo/&`
    }
}

// Bar is not region parameterized.

trait bar {
    fn any_int(&self) -> &int;
}

struct with_bar {
    f: bar
}

trait set_foo_bar {
    fn set_foo(&mut self, f: bar);
}

impl set_foo_bar for with_bar {
    fn set_foo(&mut self, f: bar) {
        self.f = f;
    }
}

fn main() {}
