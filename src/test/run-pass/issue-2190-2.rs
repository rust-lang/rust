// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test FIXME #2190
mod a {
    fn foo(f: &fn()) { f() }
    fn bar() {}
    pub fn main() { foo(||bar()); }
}

mod b {
    fn foo(f: Option<&fn()>) { f.iter(|x|x()) }
    fn bar() {}
    pub fn main() { foo(Some(bar)); }
}

mod c {
    fn foo(f: Option<&fn()>) { f.iter(|x|x()) }
    fn bar() {}
    pub fn main() { foo(Some(||bar())); }
}

pub fn main() {
}
