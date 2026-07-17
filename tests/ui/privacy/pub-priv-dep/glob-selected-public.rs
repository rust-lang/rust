//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:reexport=reexport.rs
//@ compile-flags: -Zunstable-options
//@ check-pass

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate reexport;
extern crate shared;

mod through_public {
    pub use crate::reexport::Shared as Selected;
}

mod through_private {
    pub use crate::shared::Shared as Selected;
}

use through_public::*;
use through_private::*;

pub fn selected() -> Selected {
    loop {}
}
