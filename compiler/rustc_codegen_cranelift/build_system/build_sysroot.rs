use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use crate::path::{Dirs, RelPath};
use crate::prepare::apply_patches;
use crate::rustc_info::{get_default_sysroot, get_file_name};
use crate::utils::{
    CargoProject, Compiler, LogGroup, ensure_empty_dir, spawn_and_wait, try_hard_link,
};
use crate::{CodegenBackend, SysrootKind, config};

pub(crate) fn build_sysroot(
    dirs: &Dirs,
    sysroot_kind: SysrootKind,
    cg_clif_dylib_src: &CodegenBackend,
    bootstrap_host_compiler: &Compiler,
    rustup_toolchain_name: Option<&str>,
    target_triple: String,
) -> Compiler {
    let _guard = LogGroup::guard("Build sysroot");

    eprintln!("[BUILD] sysroot {:?}", sysroot_kind);

    let dist_dir = &dirs.dist_dir;

    ensure_empty_dir(dist_dir);
    fs::create_dir_all(dist_dir.join("bin")).unwrap();
    fs::create_dir_all(dist_dir.join("lib")).unwrap();

    let is_native = bootstrap_host_compiler.triple == target_triple;

    let cg_clif_dylib_path = match cg_clif_dylib_src {
        CodegenBackend::Local(src_path) => {
            // Copy the backend
            let cg_clif_dylib_path = dist_dir.join("lib").join(src_path.file_name().unwrap());
            try_hard_link(src_path, &cg_clif_dylib_path);
            CodegenBackend::Local(cg_clif_dylib_path)
        }
        CodegenBackend::Builtin(name) => CodegenBackend::Builtin(name.clone()),
    };

    // Build and copy rustc and cargo wrappers
    let wrapper_base_name = get_file_name(&bootstrap_host_compiler.rustc, "____", "bin");
    for wrapper in ["rustc-clif", "rustdoc-clif", "cargo-clif"] {
        let wrapper_name = wrapper_base_name.replace("____", wrapper);

        let mut build_cargo_wrapper_cmd = Command::new(&bootstrap_host_compiler.rustc);
        let wrapper_path = dist_dir.join(&wrapper_name);
        build_cargo_wrapper_cmd
            .arg(dirs.source_dir.join("scripts").join(&format!("{wrapper}.rs")))
            .arg("-o")
            .arg(&wrapper_path)
            .arg("-Cstrip=debuginfo");
        if let Some(rustup_toolchain_name) = &rustup_toolchain_name {
            build_cargo_wrapper_cmd
                .env("TOOLCHAIN_NAME", rustup_toolchain_name)
                .env_remove("CARGO")
                .env_remove("RUSTC")
                .env_remove("RUSTDOC");
        } else {
            build_cargo_wrapper_cmd
                .env_remove("TOOLCHAIN_NAME")
                .env("CARGO", &bootstrap_host_compiler.cargo)
                .env("RUSTC", &bootstrap_host_compiler.rustc)
                .env("RUSTDOC", &bootstrap_host_compiler.rustdoc);
        }
        if let CodegenBackend::Builtin(name) = cg_clif_dylib_src {
            build_cargo_wrapper_cmd.env("BUILTIN_BACKEND", name);
        }
        spawn_and_wait(build_cargo_wrapper_cmd);
        try_hard_link(wrapper_path, dist_dir.join("bin").join(wrapper_name));
    }

    let host = build_sysroot_for_triple(
        dirs,
        bootstrap_host_compiler.clone(),
        &cg_clif_dylib_path,
        sysroot_kind,
    );
    host.install_into_sysroot(dist_dir);

    if !is_native {
        build_sysroot_for_triple(
            dirs,
            {
                let mut bootstrap_target_compiler = bootstrap_host_compiler.clone();
                bootstrap_target_compiler.triple = target_triple.clone();
                bootstrap_target_compiler.set_cross_linker_and_runner();
                bootstrap_target_compiler
            },
            &cg_clif_dylib_path,
            sysroot_kind,
        )
        .install_into_sysroot(dist_dir);
    }

    let mut target_compiler = Compiler {
        cargo: bootstrap_host_compiler.cargo.clone(),
        rustc: dist_dir.join(wrapper_base_name.replace("____", "rustc-clif")),
        rustdoc: dist_dir.join(wrapper_base_name.replace("____", "rustdoc-clif")),
        rustflags: vec![],
        rustdocflags: vec![],
        triple: target_triple,
        runner: vec![],
    };
    if !is_native {
        target_compiler.set_cross_linker_and_runner();
    }
    target_compiler
}

#[must_use]
struct SysrootTarget {
    triple: String,
    libs: Vec<PathBuf>,
}

impl SysrootTarget {
    fn install_into_sysroot(&self, sysroot: &Path) {
        if self.libs.is_empty() {
            return;
        }

        let target_rustlib_lib = sysroot.join("lib").join("rustlib").join(&self.triple).join("lib");
        fs::create_dir_all(&target_rustlib_lib).unwrap();

        for lib in &self.libs {
            try_hard_link(lib, target_rustlib_lib.join(lib.file_name().unwrap()));
        }
    }
}

static STDLIB_SRC: RelPath = RelPath::build("stdlib");
static STANDARD_LIBRARY: CargoProject =
    CargoProject::new(&RelPath::build("stdlib/library/sysroot"), "stdlib_target");
static RTSTARTUP_SYSROOT: RelPath = RelPath::build("rtstartup");

fn build_sysroot_for_triple(
    dirs: &Dirs,
    compiler: Compiler,
    cg_clif_dylib_path: &CodegenBackend,
    sysroot_kind: SysrootKind,
) -> SysrootTarget {
    match sysroot_kind {
        SysrootKind::None => build_rtstartup(dirs, &compiler)
            .unwrap_or(SysrootTarget { triple: compiler.triple, libs: vec![] }),
        SysrootKind::Llvm => build_llvm_sysroot_for_triple(compiler),
        SysrootKind::Clif => build_clif_sysroot_for_triple(dirs, compiler, cg_clif_dylib_path),
    }
}

fn build_llvm_sysroot_for_triple(compiler: Compiler) -> SysrootTarget {
    let default_sysroot = crate::rustc_info::get_default_sysroot(&compiler.rustc);

    let mut target_libs = SysrootTarget { triple: compiler.triple, libs: vec![] };

    for entry in fs::read_dir(
        default_sysroot.join("lib").join("rustlib").join(&target_libs.triple).join("lib"),
    )
    .unwrap()
    {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            continue;
        }
        let file = entry.path();
        let file_name_str = file.file_name().unwrap().to_str().unwrap();
        if (file_name_str.contains("rustc_")
            && !file_name_str.contains("rustc_std_workspace_")
            && !file_name_str.contains("rustc_demangle")
            && !file_name_str.contains("rustc_literal_escaper"))
            || file_name_str.contains("chalk")
            || file_name_str.contains("tracing")
            || file_name_str.contains("regex")
        {
            // These are large crates that are part of the rustc-dev component and are not
            // necessary to run regular programs.
            continue;
        }
        target_libs.libs.push(file);
    }

    target_libs
}

fn build_clif_sysroot_for_triple(
    dirs: &Dirs,
    mut compiler: Compiler,
    cg_clif_dylib_path: &CodegenBackend,
) -> SysrootTarget {
    let mut target_libs = SysrootTarget { triple: compiler.triple.clone(), libs: vec![] };

    if let Some(rtstartup_target_libs) = build_rtstartup(dirs, &compiler) {
        rtstartup_target_libs.install_into_sysroot(&RTSTARTUP_SYSROOT.to_path(dirs));

        target_libs.libs.extend(rtstartup_target_libs.libs);
    }

    let build_dir = STANDARD_LIBRARY.target_dir(dirs).join(&compiler.triple).join("release");

    if !config::get_bool("keep_sysroot") {
        // Cleanup the deps dir, but keep build scripts and the incremental cache for faster
        // recompilation as they are not affected by changes in cg_clif.
        ensure_empty_dir(&build_dir.join("deps"));
    }

    // Build sysroot
    let mut rustflags = vec!["-Zforce-unstable-if-unmarked".to_owned(), "-Cpanic=abort".to_owned()];
    match cg_clif_dylib_path {
        CodegenBackend::Local(path) => {
            rustflags.push(format!("-Zcodegen-backend={}", path.to_str().unwrap()));
        }
        CodegenBackend::Builtin(name) => {
            rustflags.push(format!("-Zcodegen-backend={name}"));
        }
    };
    // Necessary for MinGW to find rsbegin.o and rsend.o
    rustflags.push("--sysroot".to_owned());
    rustflags.push(RTSTARTUP_SYSROOT.to_path(dirs).to_str().unwrap().to_owned());

    // Incremental compilation by default disables mir inlining. This leads to both a decent
    // compile perf and a significant runtime perf regression. As such forcefully enable mir
    // inlining.
    rustflags.push("-Zinline-mir".to_owned());

    if let Some(prefix) = env::var_os("CG_CLIF_STDLIB_REMAP_PATH_PREFIX") {
        rustflags.push("--remap-path-prefix".to_owned());
        rustflags.push(format!(
            "{}={}",
            STDLIB_SRC.to_path(dirs).to_str().unwrap(),
            prefix.to_str().unwrap()
        ));
    }
    compiler.rustflags.extend(rustflags);
    let mut build_cmd = STANDARD_LIBRARY.build(&compiler, dirs);
    build_cmd.arg("--release");
    build_cmd.arg("--features").arg("backtrace panic-unwind");
    build_cmd.arg(format!("-Zroot-dir={}", STDLIB_SRC.to_path(dirs).display()));
    build_cmd.env("CARGO_PROFILE_RELEASE_DEBUG", "true");
    build_cmd.env("__CARGO_DEFAULT_LIB_METADATA", "cg_clif");
    if compiler.triple.contains("apple") {
        build_cmd.env("CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO", "packed");
    }
    spawn_and_wait(build_cmd);

    for entry in fs::read_dir(build_dir.join("deps")).unwrap() {
        let entry = entry.unwrap();
        if let Some(ext) = entry.path().extension() {
            if ext == "rmeta" || ext == "d" || ext == "dSYM" || ext == "clif" {
                continue;
            }
        } else {
            continue;
        };
        target_libs.libs.push(entry.path());
    }

    target_libs
}

fn build_rtstartup(dirs: &Dirs, compiler: &Compiler) -> Option<SysrootTarget> {
    if !config::get_bool("keep_sysroot") {
        let sysroot_src_orig = get_default_sysroot(&compiler.rustc).join("lib/rustlib/src/rust");
        assert!(sysroot_src_orig.exists());

        apply_patches(dirs, "stdlib", &sysroot_src_orig, &STDLIB_SRC.to_path(dirs));
    }

    if !compiler.triple.ends_with("windows-gnu") {
        return None;
    }

    let rtstartup_sysroot = RTSTARTUP_SYSROOT.to_path(dirs);
    ensure_empty_dir(&rtstartup_sysroot);

    let rtstartup_src = STDLIB_SRC.to_path(dirs).join("library").join("rtstartup");
    let mut target_libs = SysrootTarget { triple: compiler.triple.clone(), libs: vec![] };

    for file in ["rsbegin", "rsend"] {
        let obj = rtstartup_sysroot.join(format!("{file}.o"));
        let mut build_rtstartup_cmd = Command::new(&compiler.rustc);
        build_rtstartup_cmd
            .arg("--target")
            .arg(&compiler.triple)
            .arg("--emit=obj")
            .arg("-o")
            .arg(&obj)
            .arg(rtstartup_src.join(format!("{file}.rs")));
        spawn_and_wait(build_rtstartup_cmd);
        target_libs.libs.push(obj.clone());
    }

    Some(target_libs)
}
