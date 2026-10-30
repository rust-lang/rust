//@check-pass
//@aux-build:../auxiliary/proc_macros.rs

#![warn(clippy::missing_safety_doc)]

extern crate proc_macros;
use proc_macros::external;

#[doc = concat!(
    "Explodes.\n",
    "# Safety\n",
    "Some descriptions",
)]
pub unsafe fn with_concat_macro_doc() {}

// clippy won't recognize "safety" from external
#[doc = external!{ concat!(
    "Explodes.\n",
    "# Safety\n",
    "Some descriptions",
)}]
pub unsafe fn in_external_macro1() {}

#[doc = concat!("## Safety\n", "Some descriptions", external! {"Something from external"})]
pub unsafe fn in_external_macro2() {}

fn main() {}
