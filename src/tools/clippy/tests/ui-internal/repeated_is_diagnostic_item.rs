#![feature(rustc_private)]

extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_middle;
extern crate rustc_span;

use clippy_utils::res::MaybeDef;
use clippy_utils::sym;
use rustc_hir::def_id::DefId;
use rustc_lint::LateContext;
use rustc_middle::ty::{AdtDef, Ty, TyCtxt};
use rustc_span::Symbol;

fn binops(cx: &LateContext<'_>, ty: Ty<'_>, adt_def: &AdtDef<'_>) {
    let did = ty.opt_def_id().unwrap();

    let _ = ty.is_diag_item(cx, sym::Option) || ty.is_diag_item(cx, sym::Result);
    //~^ repeated_is_diagnostic_item
    let _ = !ty.is_diag_item(cx, sym::Option) && !ty.is_diag_item(cx, sym::Result);
    //~^ repeated_is_diagnostic_item
    let _ = adt_def.is_diag_item(cx, sym::Option) || adt_def.is_diag_item(cx, sym::Result);
    //~^ repeated_is_diagnostic_item
    let _ = !adt_def.is_diag_item(cx, sym::Option) && !adt_def.is_diag_item(cx, sym::Result);
    //~^ repeated_is_diagnostic_item
    let _ = cx.tcx.is_diagnostic_item(sym::Option, did) || cx.tcx.is_diagnostic_item(sym::Result, did);
    //~^ repeated_is_diagnostic_item
    let _ = !cx.tcx.is_diagnostic_item(sym::Option, did) && !cx.tcx.is_diagnostic_item(sym::Result, did);
    //~^ repeated_is_diagnostic_item

    // Don't lint: `is_diagnostic_item` is called not on `TyCtxt`
    struct FakeTyCtxt;
    impl FakeTyCtxt {
        fn is_diagnostic_item(&self, sym: Symbol, did: DefId) -> bool {
            unimplemented!()
        }
    }
    let f = FakeTyCtxt;
    let _ = f.is_diagnostic_item(sym::Option, did) || f.is_diagnostic_item(sym::Result, did);

    // Don't lint: `is_diagnostic_item` on `TyCtxt` comes from a(n unrelated) trait
    trait IsDiagnosticItem {
        fn is_diagnostic_item(&self, sym: Symbol, did: DefId) -> bool;
    }
    impl IsDiagnosticItem for TyCtxt<'_> {
        fn is_diagnostic_item(&self, sym: Symbol, did: DefId) -> bool {
            unimplemented!()
        }
    }
    let _ = IsDiagnosticItem::is_diagnostic_item(&cx.tcx, sym::Option, did)
        || IsDiagnosticItem::is_diagnostic_item(&cx.tcx, sym::Result, did);

    // Don't lint: `is_diag_item` is an inherent method
    struct DoesntImplMaybeDef;
    impl DoesntImplMaybeDef {
        fn is_diag_item(&self, cx: &LateContext, sym: Symbol) -> bool {
            unimplemented!()
        }
    }
    let d = DoesntImplMaybeDef;
    let _ = d.is_diag_item(cx, sym::Option) || d.is_diag_item(cx, sym::Result);

    // Don't lint: `is_diag_item` comes from a trait other than `MaybeDef`
    trait FakeMaybeDef {
        fn is_diag_item(&self, cx: &LateContext, sym: Symbol) -> bool;
    }
    struct Bar;
    impl FakeMaybeDef for Bar {
        fn is_diag_item(&self, cx: &LateContext, sym: Symbol) -> bool {
            unimplemented!()
        }
    }
    let b = Bar;
    let _ = b.is_diag_item(cx, sym::Option) || b.is_diag_item(cx, sym::Result);
}

fn main() {}
