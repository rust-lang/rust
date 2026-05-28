use std::ffi::OsStr;
use std::path::Path;

use crate::utils::run_command_with_output;

fn show_usage() {
    println!(
        r#"
`abi-test` command help:
    --help                 : Show this help"#
    );
}

pub fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(2);
    // FractalFir: In the future, I'd like to add some more subcommands / options.
    // So, this loop ought to stay for that purpose. It should also stay as a while loop(to parse args)
    #[allow(clippy::never_loop, clippy::while_let_on_iterator)]
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" => {
                show_usage();
                return Ok(());
            }
            _ => return Err(format!("Unknown option {arg:?}")),
        }
    }
    // Ensure that we have a cloned version of abi-cafe on hand.
    crate::utils::git_clone(
        "https://github.com/Gankra/abi-cafe.git",
        Some("clones/abi-cafe".as_ref()),
        true,
    )
    .map_err(|err| format!("Git clone failed with message: {err:?}!"))?;
    // Configure abi-cafe to use the exact same rustc version we use - this is crucial.
    // Otherwise, the concept of ABI compatibility becomes meanignless.
    std::fs::copy("rust-toolchain", "clones/abi-cafe/rust-toolchain")
        .expect("Could not copy toolchain configs!");
    // Get the backend path.
    // We will use the *debug* build of the backend - it has more checks enabled.
    let backend_path = std::path::absolute("target/debug/librustc_codegen_gcc.so").unwrap();
    let backend_arg = format!("--add-rustc-codegen-backend=cg_gcc:{}", backend_path.display());
    // Run ABI cafe using cargo.
    let cmd: &[&dyn AsRef<OsStr>] = &[
        &"cargo",
        &"run",
        &"--release",
        &"--",
        &backend_arg,
        // Test rust-LLVM to Rust-GCC calls
        &"--pairs",
        &"rustc_calls_cg_gcc",
        &"--pairs",
        &"cg_gcc_calls_rustc",
        // Test Rust-GCC to C calls
        &"--pairs",
        &"cg_gcc_calls_c",
        &"--pairs",
        &"c_calls_cg_gcc",
    ];
    // Run ABI cafe.
    run_command_with_output(cmd, Some(Path::new("clones/abi-cafe")))?;

    Ok(())
}
