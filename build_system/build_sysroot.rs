use std::fs;
use std::path::Path;
use std::process::{self, Command};

use super::path::{Dirs, RelPath};
use super::rustc_info::{
    get_file_name, get_rustc_version, get_toolchain_name, get_wrapper_file_name,
};
use super::utils::{spawn_and_wait, try_hard_link, CargoProject, Compiler};
use super::SysrootKind;

static DIST_DIR: RelPath = RelPath::DIST;
static BIN_DIR: RelPath = RelPath::DIST.join("bin");
static LIB_DIR: RelPath = RelPath::DIST.join("lib");
static RUSTLIB_DIR: RelPath = LIB_DIR.join("rustlib");

pub(crate) fn build_sysroot(
    dirs: &Dirs,
    channel: &str,
    sysroot_kind: SysrootKind,
    cg_clif_dylib_src: &Path,
    bootstrap_host_compiler: &Compiler,
    target_triple: String,
) -> Compiler {
    eprintln!("[BUILD] sysroot {:?}", sysroot_kind);

    DIST_DIR.ensure_fresh(dirs);
    BIN_DIR.ensure_exists(dirs);
    LIB_DIR.ensure_exists(dirs);

    let is_native = bootstrap_host_compiler.triple == target_triple;

    // Copy the backend
    let cg_clif_dylib_path = if cfg!(windows) {
        // Windows doesn't have rpath support, so the cg_clif dylib needs to be next to the
        // binaries.
        BIN_DIR
    } else {
        LIB_DIR
    }
    .to_path(dirs)
    .join(get_file_name("rustc_codegen_cranelift", "dylib"));
    try_hard_link(cg_clif_dylib_src, &cg_clif_dylib_path);

    // Build and copy rustc and cargo wrappers
    for wrapper in ["rustc-clif", "rustdoc-clif", "cargo-clif"] {
        let wrapper_name = get_wrapper_file_name(wrapper, "bin");

        let mut build_cargo_wrapper_cmd = Command::new(&bootstrap_host_compiler.rustc);
        build_cargo_wrapper_cmd
            .env("TOOLCHAIN_NAME", get_toolchain_name())
            .arg(RelPath::SCRIPTS.to_path(dirs).join(&format!("{wrapper}.rs")))
            .arg("-o")
            .arg(DIST_DIR.to_path(dirs).join(wrapper_name))
            .arg("-g");
        spawn_and_wait(build_cargo_wrapper_cmd);
    }

    let default_sysroot = super::rustc_info::get_default_sysroot(&bootstrap_host_compiler.rustc);

    let host_rustlib_lib =
        RUSTLIB_DIR.to_path(dirs).join(&bootstrap_host_compiler.triple).join("lib");
    let target_rustlib_lib = RUSTLIB_DIR.to_path(dirs).join(&target_triple).join("lib");
    fs::create_dir_all(&host_rustlib_lib).unwrap();
    fs::create_dir_all(&target_rustlib_lib).unwrap();

    if target_triple.ends_with("windows-gnu") {
        eprintln!("[BUILD] rtstartup for {target_triple}");

        let rtstartup_src = SYSROOT_SRC.to_path(dirs).join("library").join("rtstartup");

        for file in ["rsbegin", "rsend"] {
            let mut build_rtstartup_cmd = Command::new(&bootstrap_host_compiler.rustc);
            build_rtstartup_cmd
                .arg("--target")
                .arg(&target_triple)
                .arg("--emit=obj")
                .arg("-o")
                .arg(target_rustlib_lib.join(format!("{file}.o")))
                .arg(rtstartup_src.join(format!("{file}.rs")));
            spawn_and_wait(build_rtstartup_cmd);
        }
    }

    match sysroot_kind {
        SysrootKind::None => {} // Nothing to do
        SysrootKind::Llvm => {
            for file in fs::read_dir(
                default_sysroot
                    .join("lib")
                    .join("rustlib")
                    .join(&bootstrap_host_compiler.triple)
                    .join("lib"),
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

            if !is_native {
                for file in fs::read_dir(
                    default_sysroot.join("lib").join("rustlib").join(&target_triple).join("lib"),
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
                dirs,
                channel,
                bootstrap_host_compiler.clone(),
                &cg_clif_dylib_path,
            );

            if !is_native {
                build_clif_sysroot_for_triple(
                    dirs,
                    channel,
                    {
                        let mut bootstrap_target_compiler = bootstrap_host_compiler.clone();
                        bootstrap_target_compiler.triple = target_triple.clone();
                        bootstrap_target_compiler.set_cross_linker_and_runner();
                        bootstrap_target_compiler
                    },
                    &cg_clif_dylib_path,
                );
            }

            // Copy std for the host to the lib dir. This is necessary for the jit mode to find
            // libstd.
            for file in fs::read_dir(host_rustlib_lib).unwrap() {
                let file = file.unwrap().path();
                let filename = file.file_name().unwrap().to_str().unwrap();
                if filename.contains("std-") && !filename.contains(".rlib") {
                    try_hard_link(&file, LIB_DIR.to_path(dirs).join(file.file_name().unwrap()));
                }
            }
        }
    }

    let mut target_compiler = Compiler::clif_with_triple(&dirs, target_triple);
    if !is_native {
        target_compiler.set_cross_linker_and_runner();
    }
    target_compiler
}

pub(crate) static ORIG_BUILD_SYSROOT: RelPath = RelPath::SOURCE.join("build_sysroot");
pub(crate) static BUILD_SYSROOT: RelPath = RelPath::DOWNLOAD.join("sysroot");
pub(crate) static SYSROOT_RUSTC_VERSION: RelPath = BUILD_SYSROOT.join("rustc_version");
pub(crate) static SYSROOT_SRC: RelPath = BUILD_SYSROOT.join("sysroot_src");
pub(crate) static STANDARD_LIBRARY: CargoProject =
    CargoProject::new(&BUILD_SYSROOT, "build_sysroot");

fn build_clif_sysroot_for_triple(
    dirs: &Dirs,
    channel: &str,
    mut compiler: Compiler,
    cg_clif_dylib_path: &Path,
) {
    match fs::read_to_string(SYSROOT_RUSTC_VERSION.to_path(dirs)) {
        Err(e) => {
            eprintln!("Failed to get rustc version for patched sysroot source: {}", e);
            eprintln!("Hint: Try `./y.rs prepare` to patch the sysroot source");
            process::exit(1);
        }
        Ok(source_version) => {
            let rustc_version = get_rustc_version(&compiler.rustc);
            if source_version != rustc_version {
                eprintln!("The patched sysroot source is outdated");
                eprintln!("Source version: {}", source_version.trim());
                eprintln!("Rustc version:  {}", rustc_version.trim());
                eprintln!("Hint: Try `./y.rs prepare` to update the patched sysroot source");
                process::exit(1);
            }
        }
    }

    let build_dir = STANDARD_LIBRARY.target_dir(dirs).join(&compiler.triple).join(channel);

    if !super::config::get_bool("keep_sysroot") {
        // Cleanup the deps dir, but keep build scripts and the incremental cache for faster
        // recompilation as they are not affected by changes in cg_clif.
        if build_dir.join("deps").exists() {
            fs::remove_dir_all(build_dir.join("deps")).unwrap();
        }
    }

    // Build sysroot
    let mut rustflags = " -Zforce-unstable-if-unmarked -Cpanic=abort".to_string();
    rustflags.push_str(&format!(" -Zcodegen-backend={}", cg_clif_dylib_path.to_str().unwrap()));
    // Necessary for MinGW to find rsbegin.o and rsend.o
    rustflags.push_str(&format!(" --sysroot={}", DIST_DIR.to_path(dirs).to_str().unwrap()));
    if channel == "release" {
        rustflags.push_str(" -Zmir-opt-level=3");
    }
    compiler.rustflags += &rustflags;
    let mut build_cmd = STANDARD_LIBRARY.build(&compiler, dirs);
    if channel == "release" {
        build_cmd.arg("--release");
    }
    build_cmd.env("__CARGO_DEFAULT_LIB_METADATA", "cg_clif");
    spawn_and_wait(build_cmd);

    // Copy all relevant files to the sysroot
    for entry in fs::read_dir(build_dir.join("deps")).unwrap() {
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
            RUSTLIB_DIR.to_path(dirs).join(&compiler.triple).join("lib").join(entry.file_name()),
        );
    }
}
