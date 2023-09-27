// run-pass
// Test StableMIR behavior when different results are given

// ignore-stage1
// ignore-cross-compile
// ignore-remote
// edition: 2021

#![feature(rustc_private)]
#![feature(assert_matches)]

extern crate rustc_middle;
extern crate rustc_smir;
extern crate stable_mir;

use rustc_middle::ty::TyCtxt;
use rustc_smir::rustc_internal;
use std::io::Write;
use std::ops::ControlFlow;

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
    let continue_fn = |_: TyCtxt| ControlFlow::Continue::<(), bool>(true);
    let result = rustc_internal::StableMir::new(args, continue_fn).run();
    assert_eq!(result, Ok(true));
}

fn test_break(args: Vec<String>) {
    let continue_fn = |_: TyCtxt| ControlFlow::Break::<bool, i32>(false);
    let result = rustc_internal::StableMir::new(args, continue_fn).run();
    assert_eq!(result, Err(stable_mir::CompilerError::Interrupted(false)));
}

fn test_skipped(mut args: Vec<String>) {
    args.push("--version".to_string());
    let unreach_fn = |_: TyCtxt| -> ControlFlow<()> { unreachable!() };
    let result = rustc_internal::StableMir::new(args, unreach_fn).run();
    assert_eq!(result, Err(stable_mir::CompilerError::Skipped));
}

fn test_failed(mut args: Vec<String>) {
    args.push("--cfg=broken".to_string());
    let unreach_fn = |_: TyCtxt| -> ControlFlow<()> { unreachable!() };
    let result = rustc_internal::StableMir::new(args, unreach_fn).run();
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
