// run-pass
// Test that users are able to use stable mir APIs to retrieve information of the current crate

// ignore-stage-1
// ignore-cross-compile
// ignore-remote

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_smir;

use rustc_driver::{Callbacks, Compilation, RunCompiler};
use rustc_hir::def::DefKind;
use rustc_interface::{interface, Queries};
use rustc_middle::ty::TyCtxt;
use rustc_smir::{rustc_internal, stable_mir};
use std::io::Write;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir(tcx: TyCtxt<'_>) {
    // Get the local crate using stable_mir API.
    let local = stable_mir::local_crate();
    assert_eq!(&local.name, CRATE_NAME);

    // Find items in the local crate.
    let items = stable_mir::all_local_items();
    assert!(has_item(tcx, &items, (DefKind::Fn, "foo_bar")));
    assert!(has_item(tcx, &items, (DefKind::Fn, "foo::bar")));

    // Find the `std` crate.
    assert!(stable_mir::find_crate("std").is_some());
}

// Use internal API to find a function in a crate.
fn has_item(tcx: TyCtxt, items: &stable_mir::CrateItems, item: (DefKind, &str)) -> bool {
    items.iter().any(|crate_item| {
        let def_id = rustc_internal::item_def_id(crate_item);
        tcx.def_kind(def_id) == item.0 && tcx.def_path_str(def_id) == item.1
    })
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// It will invoke the compiler using a custom Callback implementation, which will
/// invoke Stable MIR APIs after the compiler has finished its analysis.
fn main() {
    let path = "input.rs";
    generate_input(&path).unwrap();
    let args = vec![
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    rustc_driver::catch_fatal_errors(|| {
        RunCompiler::new(&args, &mut SMirCalls {}).run().unwrap();
    })
    .unwrap();
}

struct SMirCalls {}

impl Callbacks for SMirCalls {
    /// Called after analysis. Return value instructs the compiler whether to
    /// continue the compilation afterwards (defaults to `Compilation::Continue`)
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        queries: &'tcx Queries<'tcx>,
    ) -> Compilation {
        queries.global_ctxt().unwrap().enter(|tcx| {
            test_stable_mir(tcx);
        });
        // No need to keep going.
        Compilation::Stop
    }
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
    mod foo {{
        pub fn bar(i: i32) -> i64 {{
            i as i64
        }}
    }}

    pub fn foo_bar(x: i32, y: i32) -> i64 {{
        let x_64 = foo::bar(x);
        let y_64 = foo::bar(y);
        x_64.wrapping_add(y_64)
    }}"#
    )?;
    Ok(())
}
