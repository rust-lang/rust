use crate::utils::{get_gcc_path, get_os_name, get_rustc_host_triple};
use std::collections::HashMap;
use std::env as std_env;

pub struct ConfigInfo {
    pub target_triple: String,
    pub rustc_command: Vec<String>,
    pub run_wrapper: Option<&'static str>,
}

// Returns the beginning for the command line of rustc.
pub fn set_config(
    env: &mut HashMap<String, String>,
    test_flags: &[String],
    gcc_path: Option<&str>,
) -> Result<ConfigInfo, String> {
    env.insert("CARGO_INCREMENTAL".to_owned(), "0".to_owned());

    let gcc_path = match gcc_path {
        Some(g) => g.to_owned(),
        None => get_gcc_path()?,
    };
    env.insert("GCC_PATH".to_owned(), gcc_path.clone());

    let os_name = get_os_name()?;
    let dylib_ext = match os_name.as_str() {
        "Linux" => "so",
        "Darwin" => "dylib",
        os => return Err(format!("unsupported OS `{os}`")),
    };
    let host_triple = get_rustc_host_triple()?;
    let mut linker = None;
    let mut target_triple = host_triple.as_str();
    let mut run_wrapper = None;
    // FIXME: handle this with a command line flag?
    // let mut target_triple = "m68k-unknown-linux-gnu";

    if host_triple != target_triple {
        if target_triple == "m68k-unknown-linux-gnu" {
            target_triple = "mips-unknown-linux-gnu";
            linker = Some("-Clinker=m68k-linux-gcc");
        } else if target_triple == "aarch64-unknown-linux-gnu" {
            // We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
            linker = Some("-Clinker=aarch64-linux-gnu-gcc");
            run_wrapper = Some("qemu-aarch64 -L /usr/aarch64-linux-gnu");
        } else {
            return Err(format!("unknown non-native platform `{target_triple}`"));
        }
    }
    // Since we don't support ThinLTO, disable LTO completely when not trying to do LTO.
    // TODO(antoyo): remove when we can handle ThinLTO.
    let disable_lto_lfags = "-Clto=off";
    let current_dir = std_env::current_dir().map_err(|e| format!("`current_dir` failed: {e:?}"))?;
    let cg_backend_path = current_dir
        .join("target")
        .join(if let Some(channel) = env.get(&"CHANNEL".to_owned()) {
            channel.as_str()
        } else {
            "debug"
        })
        .join(&format!("librustc_codegen_gcc.{dylib_ext}"));
    let sysroot_path = current_dir.join("build_sysroot/sysroot");
    let mut rustflags = Vec::new();
    if let Some(cg_rustflags) = env.get(&"CG_RUSTFLAGS".to_owned()) {
        rustflags.push(cg_rustflags.clone());
    }
    if let Some(linker) = linker {
        rustflags.push(linker.to_owned());
    }
    rustflags.extend_from_slice(&[
        "-Csymbol-mangling-version=v0".to_owned(),
        "-Cdebuginfo=2".to_owned(),
        disable_lto_lfags.to_owned(),
        format!("-Zcodegen-backend={}", cg_backend_path.display()),
        "--sysroot".to_owned(),
        format!("{}", sysroot_path.display()),
    ]);
    rustflags.extend_from_slice(test_flags);
    // FIXME(antoyo): remove once the atomic shim is gone
    if os_name == "Darwin" {
        rustflags.extend_from_slice(&[
            "-Clink-arg=-undefined".to_owned(),
            "-Clink-arg=dynamic_lookup".to_owned(),
        ]);
    }
    env.insert("RUSTFLAGS".to_owned(), rustflags.join(" "));
    // display metadata load errors
    env.insert("RUSTC_LOG".to_owned(), "warn".to_owned());

    let ld_library_path = format!(
        "{target}:{sysroot}:{gcc_path}",
        target = current_dir.join("target/out").display(),
        sysroot = current_dir
            .join(&format!(
                "build_sysroot/sysroot/lib/rustlib/{target_triple}/lib"
            ),)
            .display(),
    );
    env.insert("LD_LIBRARY_PATH".to_owned(), ld_library_path.clone());
    env.insert("DYLD_LIBRARY_PATH".to_owned(), ld_library_path);

    // NOTE: To avoid the -fno-inline errors, use /opt/gcc/bin/gcc instead of cc.
    // To do so, add a symlink for cc to /opt/gcc/bin/gcc in our PATH.
    // Another option would be to add the following Rust flag: -Clinker=/opt/gcc/bin/gcc
    let path = std::env::var("PATH").unwrap_or_default();
    env.insert("PATH".to_owned(), format!("/opt/gcc/bin:{path}"));

    let mut rustc_command = vec!["rustc".to_owned()];
    rustc_command.extend_from_slice(&rustflags);
    rustc_command.extend_from_slice(&[
        "-L".to_owned(),
        "crate=target/out".to_owned(),
        "--out-dir".to_owned(),
        "target/out".to_owned(),
    ]);
    Ok(ConfigInfo {
        target_triple: target_triple.to_owned(),
        rustc_command,
        run_wrapper,
    })
}
