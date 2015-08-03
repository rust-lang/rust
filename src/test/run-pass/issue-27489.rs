// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! stringify_passthrough {
    ($t:item) => {
        pub const TEST: &'static str = stringify!($t);
    }
}

mod foo {
    stringify_passthrough! {
        /// Hello world
        pub fn bar() { }
    }
}

mod bar {
    stringify_passthrough! {
        /// Hello # world
        pub fn bar() { }
    }
}

mod baz {
    stringify_passthrough! {
        /// ## Hello "### world #
        pub fn bar() { }
    }
}

fn main() {
    assert_eq!(foo::TEST, "#[doc = r#\" Hello world\"#]\npub fn bar() { }");
    assert_eq!(bar::TEST, "#[doc = r##\" Hello # world\"##]\npub fn bar() { }");
    assert_eq!(baz::TEST, "#[doc = r####\" ## Hello \"### world #\"####]\npub fn bar() { }");
}
