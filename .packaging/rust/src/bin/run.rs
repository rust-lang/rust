use std::process::Command;

fn run_and_printerror(command: &mut Command) {
    println!("Running: `{:?}`", command);
    match command.status() {
        Ok(status) => {
            if !status.success() {
                panic!("Failed: `{:?}` ({})", command, status);
            }
        }
        Err(error) => {
            panic!("Failed: `{:?}` ({})", command, error);
        }
    }
}

fn main() {
    let target_platform = env!("TARGET");

    let build_args = vec![
        "+enzyme",
        "-Zbuild-std",
        "rustc",
        "--target",
        target_platform,
        "--",
        "--emit=llvm-bc",
        "-g",
        "-Copt-level=3",
    ];

    let mut run1 = Command::new("cargo");
    run1.args(&build_args);
    run1.env("RUSTFLAGS", "--emit=llvm-bc");
    run1.arg("-Zno-link");
    run_and_printerror(&mut run1);

    let mut info_run = Command::new("echo");
    info_run.arg("First Compilation done, re-compiling now!");
    run_and_printerror(&mut info_run);

    let mut run2 = Command::new("cargo");
    run2.args(&build_args);
    run2.env("RUSTFLAGS", "--emit=llvm-bc");
    run_and_printerror(&mut run2);
}
