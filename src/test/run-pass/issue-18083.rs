// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// These crossed imports should resolve fine, and not block on
// each other and be reported as unresolved.

mod a {
    use b::{B};
    pub use self::inner::A;

    mod inner {
        pub struct A;
    }
}

mod b {
    use a::{A};
    pub use self::inner::B;

    mod inner {
        pub struct B;
    }
}

fn main() {}
