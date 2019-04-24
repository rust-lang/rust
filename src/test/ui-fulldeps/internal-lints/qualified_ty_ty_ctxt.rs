// compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![deny(usage_of_qualified_ty)]
#![allow(unused)]

extern crate rustc;

use rustc::ty::{self, Ty, TyCtxt};

macro_rules! qualified_macro {
    ($a:ident) => {
        fn ty_in_macro(
            ty_q: ty::Ty<'_>,
            ty: Ty<'_>,
            ty_ctxt_q: ty::TyCtxt<'_, '_, '_>,
            ty_ctxt: TyCtxt<'_, '_, '_>,
        ) {
            println!("{}", stringify!($a));
        }
    };
}

fn ty_qualified(
    ty_q: ty::Ty<'_>, //~ ERROR usage of qualified `ty::Ty<'_>`
    ty: Ty<'_>,
    ty_ctxt_q: ty::TyCtxt<'_, '_, '_>, //~ ERROR usage of qualified `ty::TyCtxt<'_, '_, '_>`
    ty_ctxt: TyCtxt<'_, '_, '_>,
) {
}


fn main() {
    qualified_macro!(a);
}
