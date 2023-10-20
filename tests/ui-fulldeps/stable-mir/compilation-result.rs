// run-pass
// Test StableMIR behavior when different results are given

// ignore-stage1
// ignore-cross-compile
// ignore-remote
// ignore-windows-gnu mingw has troubles with linking https://github.com/rust-lang/rust/pull/116837
// edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;
#[macro_use]
extern crate rustc_smir;
extern crate rustc_driver;
extern crate rustc_interface;
extern crate stable_mir;

use rustc_middle::ty::TyCtxt;
use rustc_smir::rustc_internal;
use std::io::Write;

/// This test will generate and analyze a dummy crate using the stable mir.
/// For that, it will first write the dummy crate into a file.
/// Then it will create a `StableMir` using custom arguments and then
/// it will run the compiler.
fn main() {
    let path = "input_compilation_result_test.rs";
    generate_input(&path).unwrap();
    let args = vec!["rustc".to_string(), path.to_string()];
    test_continue(args.clone());
    test_break(args.clone());
    test_failed(args.clone());
    test_skipped(args);
}

fn test_continue(args: Vec<String>) {
    let result = run!(args, ControlFlow::Continue::<(), bool>(true));
    assert_eq!(result, Ok(true));
}

fn test_break(args: Vec<String>) {
    let result = run!(args, ControlFlow::Break::<bool, i32>(false));
    assert_eq!(result, Err(stable_mir::CompilerError::Interrupted(false)));
}

#[allow(unreachable_code)]
fn test_skipped(mut args: Vec<String>) {
    args.push("--version".to_string());
    let result = run!(args, unreachable!() as ControlFlow<()>);
    assert_eq!(result, Err(stable_mir::CompilerError::Skipped));
}

#[allow(unreachable_code)]
fn test_failed(mut args: Vec<String>) {
    args.push("--cfg=broken".to_string());
    let result = run!(args, unreachable!() as ControlFlow<()>);
    assert_eq!(result, Err(stable_mir::CompilerError::CompilationFailed));
}

fn generate_input(path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    write!(
        file,
        r#"
    // This should trigger a compilation failure when enabled.
    #[cfg(broken)]
    mod broken_mod {{
        fn call_invalid() {{
            invalid_fn();
        }}
    }}

    fn main() {{}}
    "#
    )?;
    Ok(())
}
