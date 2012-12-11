// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name = "crate_method_reexport_grrrrrrr2")];

export rust;

use name_pool::add;

mod name_pool {
    #[legacy_exports];

    type name_pool = ();

    trait add {
        fn add(s: ~str);
    }

    impl name_pool: add {
        fn add(s: ~str) {
        }
    }
}

mod rust {
    #[legacy_exports];

    use name_pool::add;
    export add;
    export rt;
    export cx;

    type rt = @();

    trait cx {
        fn cx();
    }

    impl rt: cx {
        fn cx() {
        }
    }
}
