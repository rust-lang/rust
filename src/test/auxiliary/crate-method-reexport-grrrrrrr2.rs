// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];
#[crate_id="crate_method_reexport_grrrrrrr2"];
// NOTE: remove after the next snapshot
#[link(name = "crate_method_reexport_grrrrrrr2")];

pub use name_pool::add;

pub mod name_pool {
    pub type name_pool = ();

    pub trait add {
        fn add(&self, s: ~str);
    }

    impl add for name_pool {
        fn add(&self, _s: ~str) {
        }
    }
}

pub mod rust {
    pub use name_pool::add;

    pub type rt = @();

    pub trait cx {
        fn cx(&self);
    }

    impl cx for rt {
        fn cx(&self) {
        }
    }
}
