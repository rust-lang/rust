use std::env;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var("RUSTC_WRAPPER").map_or(false, |wrapper| wrapper.contains("sccache")) {
        eprintln!(
            "\x1b[1;93m=== Warning: Unsetting RUSTC_WRAPPER to prevent interference with sccache ===\x1b[0m"
        );
        env::remove_var("RUSTC_WRAPPER");
    }

    let sysroot = PathBuf::from(env::current_exe().unwrap().parent().unwrap());

    env::set_var("RUSTC", sysroot.join("bin/cg_clif".to_string() + env::consts::EXE_SUFFIX));

    let mut rustdoc_flags = env::var("RUSTDOCFLAGS").unwrap_or(String::new());
    rustdoc_flags.push_str(" -Cpanic=abort -Zpanic-abort-tests -Zcodegen-backend=");
    rustdoc_flags.push_str(
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
    rustdoc_flags.push_str(" --sysroot ");
    rustdoc_flags.push_str(sysroot.to_str().unwrap());
    env::set_var("RUSTDOCFLAGS", rustdoc_flags);

    // Ensure that the right toolchain is used
    env::set_var("RUSTUP_TOOLCHAIN", env!("RUSTUP_TOOLCHAIN"));

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
                    "-Zunstable-features".to_string(),
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
                    "-Zunstable-features".to_string(),
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
