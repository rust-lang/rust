use std::env;
use std::ffi::OsString;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let current_exe = env::current_exe().unwrap();
    let mut sysroot = current_exe.parent().unwrap();
    if sysroot.file_name().unwrap().to_str().unwrap() == "bin" {
        sysroot = sysroot.parent().unwrap();
    }

    let cg_clif_dylib_path = sysroot.join("lib").join(
        env::consts::DLL_PREFIX.to_string() + "rustc_codegen_cranelift" + env::consts::DLL_SUFFIX,
    );

    let passed_args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let mut args = vec![];
    args.push(OsString::from("-Cpanic=abort"));
    args.push(OsString::from("-Zpanic-abort-tests"));
    if let Some(name) = option_env!("BUILTIN_BACKEND") {
        args.push(OsString::from(format!("-Zcodegen-backend={name}")))
    } else {
        let mut codegen_backend_arg = OsString::from("-Zcodegen-backend=");
        codegen_backend_arg.push(cg_clif_dylib_path);
        args.push(codegen_backend_arg);
    }
    if !passed_args
        .iter()
        .any(|arg| arg == "--sysroot" || arg.to_str().is_some_and(|s| s.starts_with("--sysroot=")))
    {
        args.push(OsString::from("--sysroot"));
        args.push(OsString::from(sysroot.to_str().unwrap()));
    }
    if passed_args.is_empty() {
        // Don't pass any arguments when the user didn't pass any arguments
        // either to ensure the help message is shown.
        args.clear();
    }
    args.extend(passed_args);

    let rustdoc = if let Some(rustdoc) = option_env!("RUSTDOC") {
        rustdoc
    } else {
        // Ensure that the right toolchain is used
        env::set_var("RUSTUP_TOOLCHAIN", option_env!("TOOLCHAIN_NAME").expect("TOOLCHAIN_NAME"));
        "rustdoc"
    };

    #[cfg(unix)]
    panic!("Failed to spawn rustdoc: {}", Command::new(rustdoc).args(args).exec());

    #[cfg(not(unix))]
    std::process::exit(
        Command::new(rustdoc).args(args).spawn().unwrap().wait().unwrap().code().unwrap_or(1),
    );
}
