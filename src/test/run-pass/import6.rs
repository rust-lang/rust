// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use foo::zed;
use bar::baz;
mod foo {
    #[legacy_exports];
    mod zed {
        #[legacy_exports];
        fn baz() { debug!("baz"); }
    }
}
mod bar {
    #[legacy_exports];
    use zed::baz;
    export baz;
}
fn main() { baz(); }
