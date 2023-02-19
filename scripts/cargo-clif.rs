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
    rustflags.push_str(" --sysroot ");
    rustflags.push_str(sysroot.to_str().unwrap());
    env::set_var("RUSTFLAGS", env::var("RUSTFLAGS").unwrap_or(String::new()) + &rustflags);
    env::set_var("RUSTDOCFLAGS", env::var("RUSTDOCFLAGS").unwrap_or(String::new()) + &rustflags);

    // Ensure that the right toolchain is used
    env::set_var("RUSTUP_TOOLCHAIN", env!("TOOLCHAIN_NAME"));

    let args: Vec<_> = match env::args().nth(1).as_deref() {
        Some("jit") => {
            env::set_var(
                "RUSTFLAGS",
                env::var("RUSTFLAGS").unwrap_or(String::new()) + " -Cprefer-dynamic",
            );
            IntoIterator::into_iter(["rustc".to_string()])
                .chain(env::args().skip(2))
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
            IntoIterator::into_iter(["rustc".to_string()])
                .chain(env::args().skip(2))
                .chain([
                    "--".to_string(),
                    "-Zunstable-options".to_string(),
                    "-Cllvm-args=mode=jit-lazy".to_string(),
                ])
                .collect()
        }
        _ => env::args().skip(1).collect(),
    };

    #[cfg(unix)]
    Command::new("cargo").args(args).exec();

    #[cfg(not(unix))]
    std::process::exit(
        Command::new("cargo").args(args).spawn().unwrap().wait().unwrap().code().unwrap_or(1),
    );
}
