//@no-rustfix
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

fn main() {}

// if-chains with repeated calls on the same `ty`
fn if_chains(cx: &LateContext<'_>, ty: Ty<'_>, adt_def: &AdtDef<'_>) {
    let did = ty.opt_def_id().unwrap();

    let _ = if ty.is_diag_item(cx, sym::Option) {
        //~^ repeated_is_diagnostic_item
        "Option"
    } else if ty.is_diag_item(cx, sym::Result) {
        "Result"
    } else {
        return;
    };
    // should ideally suggest the following:
    // let _ = match ty.opt_diag_name() {
    //     Some(sym::Option) => {
    //         "Option"
    //     }
    //     Some(sym::Result) => {
    //         "Result"
    //     }
    //     _ => {
    //         return;
    //     }
    // };

    // same but in a stmt
    if ty.is_diag_item(cx, sym::Option) {
        //~^ repeated_is_diagnostic_item
        eprintln!("Option");
    } else if ty.is_diag_item(cx, sym::Result) {
        eprintln!("Result");
    }
    // should ideally suggest the following:
    // match ty.opt_diag_name() {
    //     Some(sym::Option) => {
    //         "Option"
    //     }
    //     Some(sym::Result) => {
    //         "Result"
    //     }
    //     _ => {}
    // };

    // nested conditions
    let _ = if ty.is_diag_item(cx, sym::Option) && 4 == 5 {
        //~^ repeated_is_diagnostic_item
        "Option"
    } else if ty.is_diag_item(cx, sym::Result) && 4 == 5 {
        "Result"
    } else {
        return;
    };

    let _ = if cx.tcx.is_diagnostic_item(sym::Option, did) {
        //~^ repeated_is_diagnostic_item
        "Option"
    } else if cx.tcx.is_diagnostic_item(sym::Result, did) {
        "Result"
    } else {
        return;
    };
    // should ideally suggest the following:
    // let _ = match cx.get_diagnostic_name(did) {
    //     Some(sym::Option) => {
    //         "Option"
    //     }
    //     Some(sym::Result) => {
    //         "Result"
    //     }
    //     _ => {
    //         return;
    //     }
    // };

    // same but in a stmt
    if cx.tcx.is_diagnostic_item(sym::Option, did) {
        //~^ repeated_is_diagnostic_item
        eprintln!("Option");
    } else if cx.tcx.is_diagnostic_item(sym::Result, did) {
        eprintln!("Result");
    }
    // should ideally suggest the following:
    // match cx.tcx.get_diagnostic_name(did) {
    //     Some(sym::Option) => {
    //         "Option"
    //     }
    //     Some(sym::Result) => {
    //         "Result"
    //     }
    //     _ => {}
    // };

    // nested conditions
    let _ = if cx.tcx.is_diagnostic_item(sym::Option, did) && 4 == 5 {
        //~^ repeated_is_diagnostic_item
        "Option"
    } else if cx.tcx.is_diagnostic_item(sym::Result, did) && 4 == 5 {
        "Result"
    } else {
        return;
    };
}

// if-chains with repeated calls on the same `ty`
fn consecutive_ifs(cx: &LateContext<'_>, ty: Ty<'_>, adt_def: &AdtDef<'_>) {
    let did = ty.opt_def_id().unwrap();

    {
        if ty.is_diag_item(cx, sym::Option) {
            //~^ repeated_is_diagnostic_item
            println!("Option");
        }
        if ty.is_diag_item(cx, sym::Result) {
            println!("Result");
        }
        println!("done!")
    }

    // nested conditions
    {
        if ty.is_diag_item(cx, sym::Option) && 4 == 5 {
            //~^ repeated_is_diagnostic_item
            println!("Option");
        }
        if ty.is_diag_item(cx, sym::Result) && 4 == 5 {
            println!("Result");
        }
        println!("done!")
    }

    {
        if cx.tcx.is_diagnostic_item(sym::Option, did) {
            //~^ repeated_is_diagnostic_item
            println!("Option");
        }
        if cx.tcx.is_diagnostic_item(sym::Result, did) {
            println!("Result");
        }
        println!("done!")
    }

    // nested conditions
    {
        if cx.tcx.is_diagnostic_item(sym::Option, did) && 4 == 5 {
            //~^ repeated_is_diagnostic_item
            println!("Option");
        }
        if cx.tcx.is_diagnostic_item(sym::Result, did) && 4 == 5 {
            println!("Result");
        }
        println!("done!")
    }

    // All the same, but the second if is the final expression
    {
        if ty.is_diag_item(cx, sym::Option) {
            //~^ repeated_is_diagnostic_item
            println!("Option");
        }
        if ty.is_diag_item(cx, sym::Result) {
            println!("Result");
        }
    }

    // nested conditions
    {
        if ty.is_diag_item(cx, sym::Option) && 4 == 5 {
            //~^ repeated_is_diagnostic_item
            println!("Option");
        }
        if ty.is_diag_item(cx, sym::Result) && 4 == 5 {
            println!("Result");
        }
    }

    {
        if cx.tcx.is_diagnostic_item(sym::Option, did) {
            //~^ repeated_is_diagnostic_item
            println!("Option");
        }
        if cx.tcx.is_diagnostic_item(sym::Result, did) {
            println!("Result");
        }
    }

    // nested conditions
    {
        if cx.tcx.is_diagnostic_item(sym::Option, did) && 4 == 5 {
            //~^ repeated_is_diagnostic_item
            println!("Option");
        }
        if cx.tcx.is_diagnostic_item(sym::Result, did) && 4 == 5 {
            println!("Result");
        }
    }
}
