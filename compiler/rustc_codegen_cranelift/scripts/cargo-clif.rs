use std::env;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let current_exe = env::current_exe().unwrap();
    let mut sysroot = current_exe.parent().unwrap();
    if sysroot.file_name().unwrap().to_str().unwrap() == "bin" {
        sysroot = sysroot.parent().unwrap();
    }

    let mut rustflags = String::new();
    rustflags.push_str(" -Cpanic=abort -Zpanic-abort-tests -Zcodegen-backend=");
    if let Some(name) = option_env!("BUILTIN_BACKEND") {
        rustflags.push_str(name);
    } else {
        rustflags.push_str(
            sysroot
                .join(if cfg!(windows) { "bin" } else { "lib" })
                .join(
                    env::consts::DLL_PREFIX.to_string()
                        + "rustc_codegen_cranelift"
                        + env::consts::DLL_SUFFIX,
                )
                .to_str()
                .unwrap(),
        );
    }
    rustflags.push_str(" --sysroot ");
    rustflags.push_str(sysroot.to_str().unwrap());
    env::set_var("RUSTFLAGS", env::var("RUSTFLAGS").unwrap_or(String::new()) + &rustflags);
    env::set_var("RUSTDOCFLAGS", env::var("RUSTDOCFLAGS").unwrap_or(String::new()) + &rustflags);

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
            env::set_var(
                "RUSTFLAGS",
                env::var("RUSTFLAGS").unwrap_or(String::new()) + " -Cprefer-dynamic",
            );
            args.remove(0);
            IntoIterator::into_iter(["rustc".to_string()])
                .chain(args)
                .chain([
                    "--".to_string(),
                    "-Zunstable-options".to_string(),
                    "-Cllvm-args=mode=jit".to_string(),
                ])
                .collect()
        }
        Some("lazy-jit") => {
            env::set_var(
                "RUSTFLAGS",
                env::var("RUSTFLAGS").unwrap_or(String::new()) + " -Cprefer-dynamic",
            );
            args.remove(0);
            IntoIterator::into_iter(["rustc".to_string()])
                .chain(args)
                .chain([
                    "--".to_string(),
                    "-Zunstable-options".to_string(),
                    "-Cllvm-args=mode=jit-lazy".to_string(),
                ])
                .collect()
        }
        _ => args,
    };

    #[cfg(unix)]
    panic!("Failed to spawn cargo: {}", Command::new(cargo).args(args).exec());

    #[cfg(not(unix))]
    std::process::exit(
        Command::new(cargo).args(args).spawn().unwrap().wait().unwrap().code().unwrap_or(1),
    );
}
