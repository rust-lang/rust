use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use super::rustc_info::{get_file_name, get_rustc_version, get_wrapper_file_name};
use super::utils::{spawn_and_wait, try_hard_link};
use super::SysrootKind;

pub(crate) fn build_sysroot(
    channel: &str,
    sysroot_kind: SysrootKind,
    target_dir: &Path,
    cg_clif_build_dir: &Path,
    host_triple: &str,
    target_triple: &str,
) {
    eprintln!("[BUILD] sysroot {:?}", sysroot_kind);

    if target_dir.exists() {
        fs::remove_dir_all(target_dir).unwrap();
    }
    fs::create_dir_all(target_dir.join("bin")).unwrap();
    fs::create_dir_all(target_dir.join("lib")).unwrap();

    // Copy the backend
    let cg_clif_dylib = get_file_name("rustc_codegen_cranelift", "dylib");
    let cg_clif_dylib_path = target_dir
        .join(if cfg!(windows) {
            // Windows doesn't have rpath support, so the cg_clif dylib needs to be next to the
            // binaries.
            "bin"
        } else {
            "lib"
        })
        .join(&cg_clif_dylib);
    try_hard_link(cg_clif_build_dir.join(cg_clif_dylib), &cg_clif_dylib_path);

    // Build and copy rustc and cargo wrappers
    for wrapper in ["rustc-clif", "cargo-clif"] {
        let wrapper_name = get_wrapper_file_name(wrapper, "bin");

        let mut build_cargo_wrapper_cmd = Command::new("rustc");
        build_cargo_wrapper_cmd
            .arg(PathBuf::from("scripts").join(format!("{wrapper}.rs")))
            .arg("-o")
            .arg(target_dir.join(wrapper_name))
            .arg("-g");
        spawn_and_wait(build_cargo_wrapper_cmd);
    }

    let default_sysroot = super::rustc_info::get_default_sysroot();

    let rustlib = target_dir.join("lib").join("rustlib");
    let host_rustlib_lib = rustlib.join(host_triple).join("lib");
    let target_rustlib_lib = rustlib.join(target_triple).join("lib");
    fs::create_dir_all(&host_rustlib_lib).unwrap();
    fs::create_dir_all(&target_rustlib_lib).unwrap();

    if target_triple == "x86_64-pc-windows-gnu" {
        if !default_sysroot.join("lib").join("rustlib").join(target_triple).join("lib").exists() {
            eprintln!(
                "The x86_64-pc-windows-gnu target needs to be installed first before it is possible \
                to compile a sysroot for it.",
            );
            process::exit(1);
        }
        for file in fs::read_dir(
            default_sysroot.join("lib").join("rustlib").join(target_triple).join("lib"),
        )
        .unwrap()
        {
            let file = file.unwrap().path();
            if file.extension().map_or(true, |ext| ext.to_str().unwrap() != "o") {
                continue; // only copy object files
            }
            try_hard_link(&file, target_rustlib_lib.join(file.file_name().unwrap()));
        }
    }

    match sysroot_kind {
        SysrootKind::None => {} // Nothing to do
        SysrootKind::Llvm => {
            for file in fs::read_dir(
                default_sysroot.join("lib").join("rustlib").join(host_triple).join("lib"),
            )
            .unwrap()
            {
                let file = file.unwrap().path();
                let file_name_str = file.file_name().unwrap().to_str().unwrap();
                if (file_name_str.contains("rustc_")
                    && !file_name_str.contains("rustc_std_workspace_")
                    && !file_name_str.contains("rustc_demangle"))
                    || file_name_str.contains("chalk")
                    || file_name_str.contains("tracing")
                    || file_name_str.contains("regex")
                {
                    // These are large crates that are part of the rustc-dev component and are not
                    // necessary to run regular programs.
                    continue;
                }
                try_hard_link(&file, host_rustlib_lib.join(file.file_name().unwrap()));
            }

            if target_triple != host_triple {
                for file in fs::read_dir(
                    default_sysroot.join("lib").join("rustlib").join(target_triple).join("lib"),
                )
                .unwrap()
                {
                    let file = file.unwrap().path();
                    try_hard_link(&file, target_rustlib_lib.join(file.file_name().unwrap()));
                }
            }
        }
        SysrootKind::Clif => {
            build_clif_sysroot_for_triple(
                channel,
                target_dir,
                host_triple,
                &cg_clif_dylib_path,
                None,
            );

            if host_triple != target_triple {
                // When cross-compiling it is often necessary to manually pick the right linker
                let linker = if target_triple == "aarch64-unknown-linux-gnu" {
                    Some("aarch64-linux-gnu-gcc")
                } else {
                    None
                };
                build_clif_sysroot_for_triple(
                    channel,
                    target_dir,
                    target_triple,
                    &cg_clif_dylib_path,
                    linker,
                );
            }

            // Copy std for the host to the lib dir. This is necessary for the jit mode to find
            // libstd.
            for file in fs::read_dir(host_rustlib_lib).unwrap() {
                let file = file.unwrap().path();
                let filename = file.file_name().unwrap().to_str().unwrap();
                if filename.contains("std-") && !filename.contains(".rlib") {
                    try_hard_link(&file, target_dir.join("lib").join(file.file_name().unwrap()));
                }
            }
        }
    }
}

fn build_clif_sysroot_for_triple(
    channel: &str,
    target_dir: &Path,
    triple: &str,
    cg_clif_dylib_path: &Path,
    linker: Option<&str>,
) {
    match fs::read_to_string(Path::new("build_sysroot").join("rustc_version")) {
        Err(e) => {
            eprintln!("Failed to get rustc version for patched sysroot source: {}", e);
            eprintln!("Hint: Try `./y.rs prepare` to patch the sysroot source");
            process::exit(1);
        }
        Ok(source_version) => {
            let rustc_version = get_rustc_version();
            if source_version != rustc_version {
                eprintln!("The patched sysroot source is outdated");
                eprintln!("Source version: {}", source_version.trim());
                eprintln!("Rustc version:  {}", rustc_version.trim());
                eprintln!("Hint: Try `./y.rs prepare` to update the patched sysroot source");
                process::exit(1);
            }
        }
    }

    let build_dir = Path::new("build_sysroot").join("target").join(triple).join(channel);

    if !super::config::get_bool("keep_sysroot") {
        // Cleanup the deps dir, but keep build scripts and the incremental cache for faster
        // recompilation as they are not affected by changes in cg_clif.
        if build_dir.join("deps").exists() {
            fs::remove_dir_all(build_dir.join("deps")).unwrap();
        }
    }

    // Build sysroot
    let mut build_cmd = Command::new("cargo");
    build_cmd.arg("build").arg("--target").arg(triple).current_dir("build_sysroot");
    let mut rustflags = "-Zforce-unstable-if-unmarked -Cpanic=abort".to_string();
    rustflags.push_str(&format!(" -Zcodegen-backend={}", cg_clif_dylib_path.to_str().unwrap()));
    if channel == "release" {
        build_cmd.arg("--release");
        rustflags.push_str(" -Zmir-opt-level=3");
    }
    if let Some(linker) = linker {
        use std::fmt::Write;
        write!(rustflags, " -Clinker={}", linker).unwrap();
    }
    build_cmd.env("RUSTFLAGS", rustflags);
    build_cmd.env("__CARGO_DEFAULT_LIB_METADATA", "cg_clif");
    spawn_and_wait(build_cmd);

    // Copy all relevant files to the sysroot
    for entry in
        fs::read_dir(Path::new("build_sysroot/target").join(triple).join(channel).join("deps"))
            .unwrap()
    {
        let entry = entry.unwrap();
        if let Some(ext) = entry.path().extension() {
            if ext == "rmeta" || ext == "d" || ext == "dSYM" || ext == "clif" {
                continue;
            }
        } else {
            continue;
        };
        try_hard_link(
            entry.path(),
            target_dir.join("lib").join("rustlib").join(triple).join("lib").join(entry.file_name()),
        );
    }
}
