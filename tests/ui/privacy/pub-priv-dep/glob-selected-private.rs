//@ aux-crate:priv:shared=shared.rs
//@ aux-crate:reexport=reexport.rs
//@ compile-flags: -Zunstable-options

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate reexport;
extern crate shared;

mod through_private {
    pub use crate::shared::Shared as Selected;
}

mod through_public {
    pub use crate::reexport::Shared as Selected;
}

use through_private::*;
use through_public::*;

pub fn selected() -> Selected {
    //~^ ERROR type `Shared` from private dependency 'shared' in public interface
    loop {}
}
