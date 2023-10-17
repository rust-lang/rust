// run-pass
// Test that users are able to use stable mir APIs to retrieve monomorphized instances

// ignore-stage1
// ignore-cross-compile
// ignore-remote
// edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]
#![feature(control_flow_enum)]

extern crate rustc_middle;
extern crate rustc_smir;
extern crate stable_mir;

use rustc_middle::ty::TyCtxt;

use stable_mir::*;
use rustc_smir::rustc_internal;
use std::io::Write;
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

/// This function uses the Stable MIR APIs to get information about the test crate.
fn test_stable_mir(_tcx: TyCtxt<'_>) -> ControlFlow<()> {
    let items = stable_mir::all_local_items();

    // Get all items and split generic vs monomorphic items.
    let (generic, mono) : (Vec<_>, Vec<_>) = items.into_iter().partition(|item| {
        item.requires_monomorphization()
    });
    assert_eq!(mono.len(), 3, "Expected 2 mono functions and one constant");
    assert_eq!(generic.len(), 2, "Expected 2 generic functions");

    // For all monomorphic items, get the correspondent instances.
    let instances = mono.iter().filter_map(|item| {
        mir::mono::Instance::try_from(*item).ok()
    }).collect::<Vec<mir::mono::Instance>>();
    assert_eq!(instances.len(), mono.len());

    // For all generic items, try_from should fail.
    assert!(generic.iter().all(|item| mir::mono::Instance::try_from(*item).is_err()));

    ControlFlow::Continue(())
}


/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "instance_input.rs";
    generate_input(&path).unwrap();
    let args = vec![
        "rustc".to_string(),
        "--crate-type=lib".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    rustc_internal::StableMir::new(args, test_stable_mir).run().unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
    pub fn ty_param<T>(t: &T) -> T where T: Clone {{
        t.clone()
    }}

    pub fn const_param<const LEN: usize>(a: [bool; LEN]) -> bool {{
        LEN > 0 && a[0]
    }}

    pub fn monomorphic() {{
    }}

    pub mod foo {{
        pub fn bar_mono(i: i32) -> i64 {{
            i as i64
        }}
    }}
    "#
    )?;
    Ok(())
}
