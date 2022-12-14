use std::env;
use std::ffi::OsString;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let sysroot = PathBuf::from(env::current_exe().unwrap().parent().unwrap());

    let cg_clif_dylib_path = sysroot.join(if cfg!(windows) { "bin" } else { "lib" }).join(
        env::consts::DLL_PREFIX.to_string() + "rustc_codegen_cranelift" + env::consts::DLL_SUFFIX,
    );

    let mut args = std::env::args_os().skip(1).collect::<Vec<_>>();
    args.push(OsString::from("-Cpanic=abort"));
    args.push(OsString::from("-Zpanic-abort-tests"));
    let mut codegen_backend_arg = OsString::from("-Zcodegen-backend=");
    codegen_backend_arg.push(cg_clif_dylib_path);
    args.push(codegen_backend_arg);
    if !args.contains(&OsString::from("--sysroot")) {
        args.push(OsString::from("--sysroot"));
        args.push(OsString::from(sysroot.to_str().unwrap()));
    }

    // Ensure that the right toolchain is used
    env::set_var("RUSTUP_TOOLCHAIN", env!("RUSTUP_TOOLCHAIN"));

    #[cfg(unix)]
    Command::new("rustc").args(args).exec();

    #[cfg(not(unix))]
    std::process::exit(
        Command::new("rustc").args(args).spawn().unwrap().wait().unwrap().code().unwrap_or(1),
    );
}
