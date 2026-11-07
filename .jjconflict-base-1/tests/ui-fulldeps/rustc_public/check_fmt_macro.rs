//@ run-pass
//! Test that users are able to use rustc_public to consume formatting macros.
//@ ignore-stage1
//@ ignore-cross-compile
//@ ignore-remote
//@ edition: 2021

#![feature(rustc_private)]

extern crate rustc_middle;

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_public;
use std::io::Write;
use std::ops::ControlFlow;
use rustc_public::run;

const CRATE_NAME: &str = "fmt_macro";

/// Test if we can pass the compilation.
fn test_fmt_macro() -> ControlFlow<()> {
    let entry_fn = rustc_public::entry_fn().unwrap().body().unwrap();
    for bb in &entry_fn.blocks {
        for stmt in &bb.statements {
            let _ = stmt;
        }
    }
    ControlFlow::Continue(())
}

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `RustcPublic` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "fmt_macro_input.rs";
    generate_input(&path).unwrap();
    let args = &[
        "rustc".to_string(),
        "-Cpanic=abort".to_string(),
        "--crate-name".to_string(),
        CRATE_NAME.to_string(),
        path.to_string(),
    ];
    run!(args, test_fmt_macro).unwrap();
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
        fn main() {{
            println!("hello world!");
        }}
    "#
    )?;
    Ok(())
}
