// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use foo::bar::{mod, First};
use self::bar::Second;

mod foo {
    pub use self::bar::baz::{mod};

    pub mod bar {
        pub mod baz {
            pub struct Fourth;
        }
        pub struct First;
        pub struct Second;
    }

    pub struct Third;
}

mod baz {
    use super::foo::{bar, mod};
    pub use foo::Third;
}

fn main() {
    let _ = First;
    let _ = Second;
    let _ = baz::Third;
    let _ = foo::baz::Fourth;
}
