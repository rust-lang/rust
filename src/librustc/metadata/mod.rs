// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[legacy_exports];
export encoder;
export creader;
export cstore;
export csearch;
export common;
export decoder;
export tyencode;
export tydecode;
export loader;
export filesearch;

#[legacy_exports]
mod common;
#[legacy_exports]
mod tyencode;
#[legacy_exports]
mod tydecode;
#[legacy_exports]
mod encoder;
#[legacy_exports]
mod decoder;
#[legacy_exports]
mod creader;
#[legacy_exports]
mod cstore;
#[legacy_exports]
mod csearch;
#[legacy_exports]
mod loader;
#[legacy_exports]
mod filesearch;


// Define the rustc API's that the metadata module has access to
// Over time we will reduce these dependencies and, once metadata has
// no dependencies on rustc it can move into its own crate.

mod middle {
    #[legacy_exports];
    pub use middle_::ty;
    pub use middle_::resolve;
}

mod front {
    #[legacy_exports];
}

mod back {
    #[legacy_exports];
}

mod driver {
    #[legacy_exports];
}

mod util {
    #[legacy_exports];
    pub use util_::ppaux;
}

mod lib {
    #[legacy_exports];
    pub use lib_::llvm;
}
