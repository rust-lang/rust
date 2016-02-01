// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub fn f() {}

    use foo as bar;
    pub use self::f as bar;
}

mod bar {
    use foo::bar::f as g; //~ ERROR unresolved import

    use foo as f;
    pub use foo::*;
}

use bar::f::f; //~ ERROR unresolved import
fn main() {}
