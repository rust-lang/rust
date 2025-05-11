//@ run-pass
//! Test that users are able to use serialize stable MIR constructs.

//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate serde;
extern crate serde_json;
extern crate stable_mir;

use rustc_middle::ty::TyCtxt;
use serde_json::to_string;
use stable_mir::mir::Body;
use std::io::{BufWriter, Write};
use std::ops::ControlFlow;

const CRATE_NAME: &str = "input";

fn serialize_to_json(_tcx: TyCtxt<'_>) -> ControlFlow<()> {
    let path = "output.json";
    let mut writer = BufWriter::new(std::fs::File::create(path).expect("Failed to create path"));
    let local_crate = stable_mir::local_crate();
    let items: Vec<Body> =
        stable_mir::all_local_items().iter().map(|item| item.expect_body()).collect();
    let crate_data = (local_crate.name, items);
    writer
        .write_all(to_string(&crate_data).expect("serde_json failed").as_bytes())
        .expect("JSON serialization failed");
    ControlFlow::Continue(())
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "internal_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run_with_tcx!(args, serialize_to_json).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
    pub fn main() {{
    }}
    "#
    )?;
    Ok(())
}
