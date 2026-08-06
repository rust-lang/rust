//@ compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![deny(rustc::usage_of_qualified_ty)]
#![allow(unused)]

extern crate rustc_middle;

use rustc_middle::ty::{self, Ty, TyCtxt};

macro_rules! qualified_macro {
    ($a:ident) => {
        fn ty_in_macro(
            ty_q: ty::Ty<'_>,
            ty: Ty<'_>,
            ty_ctxt_q: ty::TyCtxt<'_>,
            ty_ctxt: TyCtxt<'_>,
        ) {
            println!("{}", stringify!($a));
        }
    };
}

fn ty_qualified(
    ty_q: ty::Ty<'_>, //~ ERROR usage of qualified `ty::Ty<'_>`
    ty: Ty<'_>,
    ty_ctxt_q: ty::TyCtxt<'_>, //~ ERROR usage of qualified `ty::TyCtxt<'_>`
    ty_ctxt: TyCtxt<'_>,
) {
}


fn main() {
    qualified_macro!(a);
}
