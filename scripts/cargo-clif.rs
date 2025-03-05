use std::env;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Command;

include!("../build_system/shared_utils.rs");

fn main() {
    let current_exe = env::current_exe().unwrap();
    let mut sysroot = current_exe.parent().unwrap();
    if sysroot.file_name().unwrap().to_str().unwrap() == "bin" {
        sysroot = sysroot.parent().unwrap();
    }

    let mut rustflags = vec!["-Cpanic=abort".to_owned(), "-Zpanic-abort-tests".to_owned()];
    if let Some(name) = option_env!("BUILTIN_BACKEND") {
        rustflags.push(format!("-Zcodegen-backend={name}"));
    } else {
        let dylib = sysroot.join("lib").join(
            env::consts::DLL_PREFIX.to_string()
                + "rustc_codegen_cranelift"
                + env::consts::DLL_SUFFIX,
        );
        rustflags.push(format!("-Zcodegen-backend={}", dylib.to_str().unwrap()));
    }
    rustflags.push("--sysroot".to_owned());
    rustflags.push(sysroot.to_str().unwrap().to_owned());

    let cargo = if let Some(cargo) = option_env!("CARGO") {
        cargo
    } else {
        // Ensure that the right toolchain is used
        env::set_var("RUSTUP_TOOLCHAIN", option_env!("TOOLCHAIN_NAME").expect("TOOLCHAIN_NAME"));
        "cargo"
    };

    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.get(0).map(|arg| &**arg) == Some("clif") {
        // Avoid infinite recursion when invoking `cargo-clif` as cargo subcommand using
        // `cargo clif`.
        args.remove(0);
    }

    let args: Vec<_> = match args.get(0).map(|arg| &**arg) {
        Some("jit") => {
            rustflags.push("-Cprefer-dynamic".to_owned());
            args.remove(0);
            IntoIterator::into_iter(["rustc".to_string()])
                .chain(args)
                .chain([
                    "--".to_string(),
                    "-Zunstable-options".to_string(),
                    "-Cllvm-args=jit-mode".to_string(),
                ])
                .collect()
        }
        _ => args,
    };

    let mut cmd = Command::new(cargo);
    cmd.args(args);
    rustflags_to_cmd_env(
        &mut cmd,
        "RUSTFLAGS",
        &rustflags_from_env("RUSTFLAGS")
            .into_iter()
            .chain(rustflags.iter().map(|flag| flag.clone()))
            .collect::<Vec<_>>(),
    );
    rustflags_to_cmd_env(
        &mut cmd,
        "RUSTDOCFLAGS",
        &rustflags_from_env("RUSTDOCFLAGS")
            .into_iter()
            .chain(rustflags.iter().map(|flag| flag.clone()))
            .collect::<Vec<_>>(),
    );

    #[cfg(unix)]
    panic!("Failed to spawn cargo: {}", cmd.exec());

    #[cfg(not(unix))]
    std::process::exit(cmd.spawn().unwrap().wait().unwrap().code().unwrap_or(1));
}
