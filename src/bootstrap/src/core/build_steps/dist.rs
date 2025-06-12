//! Implementation of the various distribution aspects of the compiler.
//!
//! This module is responsible for creating tarballs of the standard library,
//! compiler, and documentation. This ends up being what we distribute to
//! everyone as well.
//!
//! No tarball is actually created literally in this file, but rather we shell
//! out to `rust-installer` still. This may one day be replaced with bits and
//! pieces of `rustup.rs`!

use std::collections::HashSet;
use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{env, fs};

use object::BinaryFormat;
use object::read::archive::ArchiveFile;
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::core::build_steps::doc::DocumentationFormat;
use crate::core::build_steps::tool::{self, Tool};
use crate::core::build_steps::vendor::{VENDOR_DIR, Vendor};
use crate::core::build_steps::{compile, llvm};
use crate::core::builder::{Builder, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::build_stamp::{self, BuildStamp};
use crate::utils::channel::{self, Info};
use crate::utils::exec::{BootstrapCommand, command};
use crate::utils::helpers::{
    exe, is_dylib, move_file, t, target_supports_cranelift_backend, timeit,
};
use crate::utils::tarball::{GeneratedTarball, OverlayKind, Tarball};
use crate::{Compiler, DependencyType, FileType, LLVM_TOOLS, Mode, trace};

pub fn pkgname(builder: &Builder<'_>, component: &str) -> String {
    format!("{}-{}", component, builder.rust_package_vers())
}

pub(crate) fn distdir(builder: &Builder<'_>) -> PathBuf {
    builder.out.join("dist")
}

pub fn tmpdir(builder: &Builder<'_>) -> PathBuf {
    builder.out.join("tmp/dist")
}

fn should_build_extended_tool(builder: &Builder<'_>, tool: &str) -> bool {
    if !builder.config.extended {
        return false;
    }
    builder.config.tools.as_ref().is_none_or(|tools| tools.contains(tool))
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Docs {
    pub host: TargetSelection,
}

impl Step for Docs {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = run.builder.config.docs;
        run.alias("rust-docs").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Docs { host: run.target });
    }

    /// Builds the `rust-docs` installer component.
    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let host = self.host;
        builder.default_doc(&[]);

        let dest = "share/doc/rust/html";

        let mut tarball = Tarball::new(builder, "rust-docs", &host.triple);
        tarball.set_product_name("Rust Documentation");
        tarball.add_bulk_dir(builder.doc_out(host), dest);
        tarball.add_file(builder.src.join("src/doc/robots.txt"), dest, FileType::Regular);
        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct JsonDocs {
    pub host: TargetSelection,
}

impl Step for JsonDocs {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = run.builder.config.docs;
        run.alias("rust-docs-json").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(JsonDocs { host: run.target });
    }

    /// Builds the `rust-docs-json` installer component.
    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let host = self.host;
        builder.ensure(crate::core::build_steps::doc::Std::new(
            builder.top_stage,
            host,
            DocumentationFormat::Json,
        ));

        let dest = "share/doc/rust/json";

        let mut tarball = Tarball::new(builder, "rust-docs-json", &host.triple);
        tarball.set_product_name("Rust Documentation In JSON Format");
        tarball.is_preview(true);
        tarball.add_bulk_dir(builder.json_doc_out(host), dest);
        Some(tarball.generate())
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustcDocs {
    pub host: TargetSelection,
}

impl Step for RustcDocs {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.alias("rustc-docs").default_condition(builder.config.compiler_docs)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustcDocs { host: run.target });
    }

    /// Builds the `rustc-docs` installer component.
    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let host = self.host;
        builder.default_doc(&[]);

        let mut tarball = Tarball::new(builder, "rustc-docs", &host.triple);
        tarball.set_product_name("Rustc Documentation");
        tarball.add_bulk_dir(builder.compiler_doc_out(host), "share/doc/rust/html/rustc");
        Some(tarball.generate())
    }
}

fn find_files(files: &[&str], path: &[PathBuf]) -> Vec<PathBuf> {
    let mut found = Vec::with_capacity(files.len());

    for file in files {
        let file_path = path.iter().map(|dir| dir.join(file)).find(|p| p.exists());

        if let Some(file_path) = file_path {
            found.push(file_path);
        } else {
            panic!("Could not find '{file}' in {path:?}");
        }
    }

    found
}

fn make_win_dist(
    rust_root: &Path,
    plat_root: &Path,
    target: TargetSelection,
    builder: &Builder<'_>,
) {
    if builder.config.dry_run() {
        return;
    }

    //Ask gcc where it keeps its stuff
    let mut cmd = command(builder.cc(target));
    cmd.arg("-print-search-dirs");
    let gcc_out = cmd.run_capture_stdout(builder).stdout();

    let mut bin_path: Vec<_> = env::split_paths(&env::var_os("PATH").unwrap_or_default()).collect();
    let mut lib_path = Vec::new();

    for line in gcc_out.lines() {
        let idx = line.find(':').unwrap();
        let key = &line[..idx];
        let trim_chars: &[_] = &[' ', '='];
        let value = env::split_paths(line[(idx + 1)..].trim_start_matches(trim_chars));

        if key == "programs" {
            bin_path.extend(value);
        } else if key == "libraries" {
            lib_path.extend(value);
        }
    }

    let compiler = if target == "i686-pc-windows-gnu" {
        "i686-w64-mingw32-gcc.exe"
    } else if target == "x86_64-pc-windows-gnu" {
        "x86_64-w64-mingw32-gcc.exe"
    } else {
        "gcc.exe"
    };
    let target_tools = [compiler, "ld.exe", "dlltool.exe", "libwinpthread-1.dll"];
    let mut rustc_dlls = vec!["libwinpthread-1.dll"];
    if target.starts_with("i686-") {
        rustc_dlls.push("libgcc_s_dw2-1.dll");
    } else {
        rustc_dlls.push("libgcc_s_seh-1.dll");
    }

    // Libraries necessary to link the windows-gnu toolchains.
    // System libraries will be preferred if they are available (see #67429).
    let target_libs = [
        //MinGW libs
        "libgcc.a",
        "libgcc_eh.a",
        "libgcc_s.a",
        "libm.a",
        "libmingw32.a",
        "libmingwex.a",
        "libstdc++.a",
        "libiconv.a",
        "libmoldname.a",
        "libpthread.a",
        // Windows import libs
        // This *should* contain only the set of libraries necessary to link the standard library,
        // however we've had problems with people accidentally depending on extra libs being here,
        // so we can't easily remove entries.
        "libadvapi32.a",
        "libbcrypt.a",
        "libcomctl32.a",
        "libcomdlg32.a",
        "libcredui.a",
        "libcrypt32.a",
        "libdbghelp.a",
        "libgdi32.a",
        "libimagehlp.a",
        "libiphlpapi.a",
        "libkernel32.a",
        "libmsimg32.a",
        "libmsvcrt.a",
        "libntdll.a",
        "libodbc32.a",
        "libole32.a",
        "liboleaut32.a",
        "libopengl32.a",
        "libpsapi.a",
        "librpcrt4.a",
        "libsecur32.a",
        "libsetupapi.a",
        "libshell32.a",
        "libsynchronization.a",
        "libuser32.a",
        "libuserenv.a",
        "libuuid.a",
        "libwinhttp.a",
        "libwinmm.a",
        "libwinspool.a",
        "libws2_32.a",
        "libwsock32.a",
    ];

    //Find mingw artifacts we want to bundle
    let target_tools = find_files(&target_tools, &bin_path);
    let rustc_dlls = find_files(&rustc_dlls, &bin_path);
    let target_libs = find_files(&target_libs, &lib_path);

    // Copy runtime dlls next to rustc.exe
    let rust_bin_dir = rust_root.join("bin/");
    fs::create_dir_all(&rust_bin_dir).expect("creating rust_bin_dir failed");
    for src in &rustc_dlls {
        builder.copy_link_to_folder(src, &rust_bin_dir);
    }

    if builder.config.lld_enabled {
        // rust-lld.exe also needs runtime dlls
        let rust_target_bin_dir = rust_root.join("lib/rustlib").join(target).join("bin");
        fs::create_dir_all(&rust_target_bin_dir).expect("creating rust_target_bin_dir failed");
        for src in &rustc_dlls {
            builder.copy_link_to_folder(src, &rust_target_bin_dir);
        }
    }

    //Copy platform tools to platform-specific bin directory
    let plat_target_bin_self_contained_dir =
        plat_root.join("lib/rustlib").join(target).join("bin/self-contained");
    fs::create_dir_all(&plat_target_bin_self_contained_dir)
        .expect("creating plat_target_bin_self_contained_dir failed");
    for src in target_tools {
        builder.copy_link_to_folder(&src, &plat_target_bin_self_contained_dir);
    }

    // Warn windows-gnu users that the bundled GCC cannot compile C files
    builder.create(
        &plat_target_bin_self_contained_dir.join("GCC-WARNING.txt"),
        "gcc.exe contained in this folder cannot be used for compiling C files - it is only \
         used as a linker. In order to be able to compile projects containing C code use \
         the GCC provided by MinGW or Cygwin.",
    );

    //Copy platform libs to platform-specific lib directory
    let plat_target_lib_self_contained_dir =
        plat_root.join("lib/rustlib").join(target).join("lib/self-contained");
    fs::create_dir_all(&plat_target_lib_self_contained_dir)
        .expect("creating plat_target_lib_self_contained_dir failed");
    for src in target_libs {
        builder.copy_link_to_folder(&src, &plat_target_lib_self_contained_dir);
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Mingw {
    pub host: TargetSelection,
}

impl Step for Mingw {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rust-mingw")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Mingw { host: run.target });
    }

    /// Builds the `rust-mingw` installer component.
    ///
    /// This contains all the bits and pieces to run the MinGW Windows targets
    /// without any extra installed software (e.g., we bundle gcc, libraries, etc).
    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let host = self.host;
        if !host.ends_with("pc-windows-gnu") || !builder.config.dist_include_mingw_linker {
            return None;
        }

        let mut tarball = Tarball::new(builder, "rust-mingw", &host.triple);
        tarball.set_product_name("Rust MinGW");

        // The first argument is a "temporary directory" which is just
        // thrown away (this contains the runtime DLLs included in the rustc package
        // above) and the second argument is where to place all the MinGW components
        // (which is what we want).
        make_win_dist(&tmpdir(builder), tarball.image_dir(), host, builder);

        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Rustc {
    pub compiler: Compiler,
}

impl Step for Rustc {
    type Output = GeneratedTarball;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rustc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder
            .ensure(Rustc { compiler: run.builder.compiler(run.builder.top_stage, run.target) });
    }

    /// Creates the `rustc` installer component.
    fn run(self, builder: &Builder<'_>) -> GeneratedTarball {
        let compiler = self.compiler;
        let host = self.compiler.host;

        let tarball = Tarball::new(builder, "rustc", &host.triple);

        // Prepare the rustc "image", what will actually end up getting installed
        prepare_image(builder, compiler, tarball.image_dir());

        // On MinGW we've got a few runtime DLL dependencies that we need to
        // include. The first argument to this script is where to put these DLLs
        // (the image we're creating), and the second argument is a junk directory
        // to ignore all other MinGW stuff the script creates.
        //
        // On 32-bit MinGW we're always including a DLL which needs some extra
        // licenses to distribute. On 64-bit MinGW we don't actually distribute
        // anything requiring us to distribute a license, but it's likely the
        // install will *also* include the rust-mingw package, which also needs
        // licenses, so to be safe we just include it here in all MinGW packages.
        if host.ends_with("pc-windows-gnu") && builder.config.dist_include_mingw_linker {
            make_win_dist(tarball.image_dir(), &tmpdir(builder), host, builder);
            tarball.add_dir(builder.src.join("src/etc/third-party"), "share/doc");
        }

        return tarball.generate();

        fn prepare_image(builder: &Builder<'_>, compiler: Compiler, image: &Path) {
            let host = compiler.host;
            let src = builder.sysroot(compiler);

            // Copy rustc binary
            t!(fs::create_dir_all(image.join("bin")));
            builder.cp_link_r(&src.join("bin"), &image.join("bin"));

            // If enabled, copy rustdoc binary
            if builder
                .config
                .tools
                .as_ref()
                .is_none_or(|tools| tools.iter().any(|tool| tool == "rustdoc"))
            {
                let rustdoc = builder.rustdoc(compiler);
                builder.install(&rustdoc, &image.join("bin"), FileType::Executable);
            }

            let ra_proc_macro_srv_compiler =
                builder.compiler_for(compiler.stage, builder.config.build, compiler.host);
            builder.ensure(compile::Rustc::new(ra_proc_macro_srv_compiler, compiler.host));

            if let Some(ra_proc_macro_srv) = builder.ensure_if_default(
                tool::RustAnalyzerProcMacroSrv {
                    compiler: ra_proc_macro_srv_compiler,
                    target: compiler.host,
                },
                builder.kind,
            ) {
                let dst = image.join("libexec");
                builder.install(&ra_proc_macro_srv.tool_path, &dst, FileType::Executable);
            }

            let libdir_relative = builder.libdir_relative(compiler);

            // Copy runtime DLLs needed by the compiler
            if libdir_relative.to_str() != Some("bin") {
                let libdir = builder.rustc_libdir(compiler);
                for entry in builder.read_dir(&libdir) {
                    if is_dylib(&entry.path()) {
                        // Don't use custom libdir here because ^lib/ will be resolved again
                        // with installer
                        builder.install(&entry.path(), &image.join("lib"), FileType::NativeLibrary);
                    }
                }
            }

            // Copy libLLVM.so to the lib dir as well, if needed. While not
            // technically needed by rustc itself it's needed by lots of other
            // components like the llvm tools and LLD. LLD is included below and
            // tools/LLDB come later, so let's just throw it in the rustc
            // component for now.
            maybe_install_llvm_runtime(builder, host, image);

            let dst_dir = image.join("lib/rustlib").join(host).join("bin");
            t!(fs::create_dir_all(&dst_dir));

            // Copy over lld if it's there
            if builder.config.lld_enabled {
                let src_dir = builder.sysroot_target_bindir(compiler, host);
                let rust_lld = exe("rust-lld", compiler.host);
                builder.copy_link(
                    &src_dir.join(&rust_lld),
                    &dst_dir.join(&rust_lld),
                    FileType::Executable,
                );
                let self_contained_lld_src_dir = src_dir.join("gcc-ld");
                let self_contained_lld_dst_dir = dst_dir.join("gcc-ld");
                t!(fs::create_dir(&self_contained_lld_dst_dir));
                for name in crate::LLD_FILE_NAMES {
                    let exe_name = exe(name, compiler.host);
                    builder.copy_link(
                        &self_contained_lld_src_dir.join(&exe_name),
                        &self_contained_lld_dst_dir.join(&exe_name),
                        FileType::Executable,
                    );
                }
            }

            if builder.config.llvm_enabled(compiler.host) && builder.config.llvm_tools_enabled {
                let src_dir = builder.sysroot_target_bindir(compiler, host);
                let llvm_objcopy = exe("llvm-objcopy", compiler.host);
                let rust_objcopy = exe("rust-objcopy", compiler.host);
                builder.copy_link(
                    &src_dir.join(&llvm_objcopy),
                    &dst_dir.join(&rust_objcopy),
                    FileType::Executable,
                );
            }

            if builder.tool_enabled("wasm-component-ld") {
                let src_dir = builder.sysroot_target_bindir(compiler, host);
                let ld = exe("wasm-component-ld", compiler.host);
                builder.copy_link(&src_dir.join(&ld), &dst_dir.join(&ld), FileType::Executable);
            }

            // Man pages
            t!(fs::create_dir_all(image.join("share/man/man1")));
            let man_src = builder.src.join("src/doc/man");
            let man_dst = image.join("share/man/man1");

            // don't use our `bootstrap::{copy_internal, cp_r}`, because those try
            // to hardlink, and we don't want to edit the source templates
            for file_entry in builder.read_dir(&man_src) {
                let page_src = file_entry.path();
                let page_dst = man_dst.join(file_entry.file_name());
                let src_text = t!(std::fs::read_to_string(&page_src));
                let new_text = src_text.replace("<INSERT VERSION HERE>", &builder.version);
                t!(std::fs::write(&page_dst, &new_text));
                t!(fs::copy(&page_src, &page_dst));
            }

            // Debugger scripts
            builder.ensure(DebuggerScripts { sysroot: image.to_owned(), host });

            // HTML copyright files
            let file_list = builder.ensure(super::run::GenerateCopyright);
            for file in file_list {
                builder.install(&file, &image.join("share/doc/rust"), FileType::Regular);
            }

            // README
            builder.install(
                &builder.src.join("README.md"),
                &image.join("share/doc/rust"),
                FileType::Regular,
            );

            // The REUSE-managed license files
            let license = |path: &Path| {
                builder.install(path, &image.join("share/doc/rust/licenses"), FileType::Regular);
            };
            for entry in t!(std::fs::read_dir(builder.src.join("LICENSES"))).flatten() {
                license(&entry.path());
            }
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DebuggerScripts {
    pub sysroot: PathBuf,
    pub host: TargetSelection,
}

impl Step for DebuggerScripts {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Copies debugger scripts for `target` into the `sysroot` specified.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let sysroot = self.sysroot;
        let dst = sysroot.join("lib/rustlib/etc");
        t!(fs::create_dir_all(&dst));
        let cp_debugger_script = |file: &str| {
            builder.install(&builder.src.join("src/etc/").join(file), &dst, FileType::Regular);
        };
        if host.contains("windows-msvc") {
            // windbg debugger scripts
            builder.install(
                &builder.src.join("src/etc/rust-windbg.cmd"),
                &sysroot.join("bin"),
                FileType::Script,
            );

            cp_debugger_script("natvis/intrinsic.natvis");
            cp_debugger_script("natvis/liballoc.natvis");
            cp_debugger_script("natvis/libcore.natvis");
            cp_debugger_script("natvis/libstd.natvis");
        }

        cp_debugger_script("rust_types.py");

        // gdb debugger scripts
        builder.install(
            &builder.src.join("src/etc/rust-gdb"),
            &sysroot.join("bin"),
            FileType::Script,
        );
        builder.install(
            &builder.src.join("src/etc/rust-gdbgui"),
            &sysroot.join("bin"),
            FileType::Script,
        );

        cp_debugger_script("gdb_load_rust_pretty_printers.py");
        cp_debugger_script("gdb_lookup.py");
        cp_debugger_script("gdb_providers.py");

        // lldb debugger scripts
        builder.install(
            &builder.src.join("src/etc/rust-lldb"),
            &sysroot.join("bin"),
            FileType::Script,
        );

        cp_debugger_script("lldb_lookup.py");
        cp_debugger_script("lldb_providers.py");
        cp_debugger_script("lldb_commands")
    }
}

fn skip_host_target_lib(builder: &Builder<'_>, compiler: Compiler) -> bool {
    // The only true set of target libraries came from the build triple, so
    // let's reduce redundant work by only producing archives from that host.
    if !builder.config.is_host_target(compiler.host) {
        builder.info("\tskipping, not a build host");
        true
    } else {
        false
    }
}

/// Check that all objects in rlibs for UEFI targets are COFF. This
/// ensures that the C compiler isn't producing ELF objects, which would
/// not link correctly with the COFF objects.
fn verify_uefi_rlib_format(builder: &Builder<'_>, target: TargetSelection, stamp: &BuildStamp) {
    if !target.ends_with("-uefi") {
        return;
    }

    for (path, _) in builder.read_stamp_file(stamp) {
        if path.extension() != Some(OsStr::new("rlib")) {
            continue;
        }

        let data = t!(fs::read(&path));
        let data = data.as_slice();
        let archive = t!(ArchiveFile::parse(data));
        for member in archive.members() {
            let member = t!(member);
            let member_data = t!(member.data(data));

            let is_coff = match object::File::parse(member_data) {
                Ok(member_file) => member_file.format() == BinaryFormat::Coff,
                Err(_) => false,
            };

            if !is_coff {
                let member_name = String::from_utf8_lossy(member.name());
                panic!("member {} in {} is not COFF", member_name, path.display());
            }
        }
    }
}

/// Copy stamped files into an image's `target/lib` directory.
fn copy_target_libs(
    builder: &Builder<'_>,
    target: TargetSelection,
    image: &Path,
    stamp: &BuildStamp,
) {
    let dst = image.join("lib/rustlib").join(target).join("lib");
    let self_contained_dst = dst.join("self-contained");
    t!(fs::create_dir_all(&dst));
    t!(fs::create_dir_all(&self_contained_dst));
    for (path, dependency_type) in builder.read_stamp_file(stamp) {
        if dependency_type == DependencyType::TargetSelfContained {
            builder.copy_link(
                &path,
                &self_contained_dst.join(path.file_name().unwrap()),
                FileType::NativeLibrary,
            );
        } else if dependency_type == DependencyType::Target || builder.config.is_host_target(target)
        {
            builder.copy_link(&path, &dst.join(path.file_name().unwrap()), FileType::NativeLibrary);
        }
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Std {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Std {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rust-std")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Std {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let compiler = self.compiler;
        let target = self.target;

        if skip_host_target_lib(builder, compiler) {
            return None;
        }

        builder.ensure(compile::Std::new(compiler, target));

        let mut tarball = Tarball::new(builder, "rust-std", &target.triple);
        tarball.include_target_in_component_name(true);

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        let stamp = build_stamp::libstd_stamp(builder, compiler_to_use, target);
        verify_uefi_rlib_format(builder, target, &stamp);
        copy_target_libs(builder, target, tarball.image_dir(), &stamp);

        Some(tarball.generate())
    }
}

/// Tarball containing the compiler that gets downloaded and used by
/// `rust.download-rustc`.
///
/// (Don't confuse this with [`RustDev`], without the `c`!)
#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct RustcDev {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustcDev {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rustc-dev")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustcDev {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let compiler = self.compiler;
        let target = self.target;
        if skip_host_target_lib(builder, compiler) {
            return None;
        }

        builder.ensure(compile::Rustc::new(compiler, target));

        let tarball = Tarball::new(builder, "rustc-dev", &target.triple);

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        let stamp = build_stamp::librustc_stamp(builder, compiler_to_use, target);
        copy_target_libs(builder, target, tarball.image_dir(), &stamp);

        let src_files = &["Cargo.lock"];
        // This is the reduced set of paths which will become the rustc-dev component
        // (essentially the compiler crates and all of their path dependencies).
        copy_src_dirs(
            builder,
            &builder.src,
            // The compiler has a path dependency on proc_macro, so make sure to include it.
            &["compiler", "library/proc_macro"],
            &[],
            &tarball.image_dir().join("lib/rustlib/rustc-src/rust"),
        );
        for file in src_files {
            tarball.add_file(
                builder.src.join(file),
                "lib/rustlib/rustc-src/rust",
                FileType::Regular,
            );
        }

        Some(tarball.generate())
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Analysis {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Analysis {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = should_build_extended_tool(run.builder, "analysis");
        run.alias("rust-analysis").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Analysis {
            // Find the actual compiler (handling the full bootstrap option) which
            // produced the save-analysis data because that data isn't copied
            // through the sysroot uplifting.
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    /// Creates a tarball of (degenerate) save-analysis metadata, if available.
    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let compiler = self.compiler;
        let target = self.target;
        if !builder.config.is_host_target(compiler.host) {
            return None;
        }

        let src = builder
            .stage_out(compiler, Mode::Std)
            .join(target)
            .join(builder.cargo_dir())
            .join("deps")
            .join("save-analysis");

        // Write a file indicating that this component has been removed.
        t!(std::fs::create_dir_all(&src));
        let mut removed = src.clone();
        removed.push("removed.json");
        let mut f = t!(std::fs::File::create(removed));
        t!(write!(f, r#"{{ "warning": "The `rust-analysis` component has been removed." }}"#));

        let mut tarball = Tarball::new(builder, "rust-analysis", &target.triple);
        tarball.include_target_in_component_name(true);
        tarball.add_dir(src, format!("lib/rustlib/{}/analysis", target.triple));
        Some(tarball.generate())
    }
}

/// Use the `builder` to make a filtered copy of `base`/X for X in (`src_dirs` - `exclude_dirs`) to
/// `dst_dir`.
fn copy_src_dirs(
    builder: &Builder<'_>,
    base: &Path,
    src_dirs: &[&str],
    exclude_dirs: &[&str],
    dst_dir: &Path,
) {
    fn filter_fn(exclude_dirs: &[&str], dir: &str, path: &Path) -> bool {
        let spath = match path.to_str() {
            Some(path) => path,
            None => return false,
        };
        if spath.ends_with('~') || spath.ends_with(".pyc") {
            return false;
        }

        const LLVM_PROJECTS: &[&str] = &[
            "llvm-project/clang",
            "llvm-project\\clang",
            "llvm-project/libunwind",
            "llvm-project\\libunwind",
            "llvm-project/lld",
            "llvm-project\\lld",
            "llvm-project/lldb",
            "llvm-project\\lldb",
            "llvm-project/llvm",
            "llvm-project\\llvm",
            "llvm-project/compiler-rt",
            "llvm-project\\compiler-rt",
            "llvm-project/cmake",
            "llvm-project\\cmake",
            "llvm-project/runtimes",
            "llvm-project\\runtimes",
        ];
        if spath.contains("llvm-project")
            && !spath.ends_with("llvm-project")
            && !LLVM_PROJECTS.iter().any(|path| spath.contains(path))
        {
            return false;
        }

        const LLVM_TEST: &[&str] = &["llvm-project/llvm/test", "llvm-project\\llvm\\test"];
        if LLVM_TEST.iter().any(|path| spath.contains(path))
            && (spath.ends_with(".ll") || spath.ends_with(".td") || spath.ends_with(".s"))
        {
            return false;
        }

        // Cargo tests use some files like `.gitignore` that we would otherwise exclude.
        const CARGO_TESTS: &[&str] = &["tools/cargo/tests", "tools\\cargo\\tests"];
        if CARGO_TESTS.iter().any(|path| spath.contains(path)) {
            return true;
        }

        let full_path = Path::new(dir).join(path);
        if exclude_dirs.iter().any(|excl| full_path == Path::new(excl)) {
            return false;
        }

        let excludes = [
            "CVS",
            "RCS",
            "SCCS",
            ".git",
            ".gitignore",
            ".gitmodules",
            ".gitattributes",
            ".cvsignore",
            ".svn",
            ".arch-ids",
            "{arch}",
            "=RELEASE-ID",
            "=meta-update",
            "=update",
            ".bzr",
            ".bzrignore",
            ".bzrtags",
            ".hg",
            ".hgignore",
            ".hgrags",
            "_darcs",
        ];
        !path.iter().map(|s| s.to_str().unwrap()).any(|s| excludes.contains(&s))
    }

    // Copy the directories using our filter
    for item in src_dirs {
        let dst = &dst_dir.join(item);
        t!(fs::create_dir_all(dst));
        builder
            .cp_link_filtered(&base.join(item), dst, &|path| filter_fn(exclude_dirs, item, path));
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Src;

impl Step for Src {
    /// The output path of the src installer tarball
    type Output = GeneratedTarball;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rust-src")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Src);
    }

    /// Creates the `rust-src` installer component
    fn run(self, builder: &Builder<'_>) -> GeneratedTarball {
        if !builder.config.dry_run() {
            builder.require_submodule("src/llvm-project", None);
        }

        let tarball = Tarball::new_targetless(builder, "rust-src");

        // A lot of tools expect the rust-src component to be entirely in this directory, so if you
        // change that (e.g. by adding another directory `lib/rustlib/src/foo` or
        // `lib/rustlib/src/rust/foo`), you will need to go around hunting for implicit assumptions
        // and fix them...
        //
        // NOTE: if you update the paths here, you also should update the "virtual" path
        // translation code in `imported_source_files` in `src/librustc_metadata/rmeta/decoder.rs`
        let dst_src = tarball.image_dir().join("lib/rustlib/src/rust");

        // This is the reduced set of paths which will become the rust-src component
        // (essentially libstd and all of its path dependencies).
        copy_src_dirs(
            builder,
            &builder.src,
            &["library", "src/llvm-project/libunwind"],
            &[
                // not needed and contains symlinks which rustup currently
                // chokes on when unpacking.
                "library/backtrace/crates",
                // these are 30MB combined and aren't necessary for building
                // the standard library.
                "library/stdarch/Cargo.toml",
                "library/stdarch/crates/stdarch-verify",
                "library/stdarch/crates/intrinsic-test",
            ],
            &dst_src,
        );

        tarball.generate()
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct PlainSourceTarball;

impl Step for PlainSourceTarball {
    /// Produces the location of the tarball generated
    type Output = GeneratedTarball;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.alias("rustc-src").default_condition(builder.config.rust_dist_src)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(PlainSourceTarball);
    }

    /// Creates the plain source tarball
    fn run(self, builder: &Builder<'_>) -> GeneratedTarball {
        // NOTE: This is a strange component in a lot of ways. It uses `src` as the target, which
        // means neither rustup nor rustup-toolchain-install-master know how to download it.
        // It also contains symbolic links, unlike other any other dist tarball.
        // It's used for distros building rustc from source in a pre-vendored environment.
        let mut tarball = Tarball::new(builder, "rustc", "src");
        tarball.permit_symlinks(true);
        let plain_dst_src = tarball.image_dir();

        // This is the set of root paths which will become part of the source package
        let src_files = [
            // tidy-alphabetical-start
            ".gitmodules",
            "bootstrap.example.toml",
            "Cargo.lock",
            "Cargo.toml",
            "configure",
            "CONTRIBUTING.md",
            "COPYRIGHT",
            "LICENSE-APACHE",
            "license-metadata.json",
            "LICENSE-MIT",
            "README.md",
            "RELEASES.md",
            "REUSE.toml",
            "x",
            "x.ps1",
            "x.py",
            // tidy-alphabetical-end
        ];
        let src_dirs = ["src", "compiler", "library", "tests", "LICENSES"];

        copy_src_dirs(
            builder,
            &builder.src,
            &src_dirs,
            &[
                // We don't currently use the GCC source code for building any official components,
                // it is very big, and has unclear licensing implications due to being GPL licensed.
                // We thus exclude it from the source tarball from now.
                "src/gcc",
            ],
            plain_dst_src,
        );
        // We keep something in src/gcc because it is a registered submodule,
        // and if it misses completely it can cause issues elsewhere
        // (see https://github.com/rust-lang/rust/issues/137332).
        // We can also let others know why is the source code missing.
        if !builder.config.dry_run() {
            builder.create_dir(&plain_dst_src.join("src/gcc"));
            t!(std::fs::write(
                plain_dst_src.join("src/gcc/notice.txt"),
                "The GCC source code is not included due to unclear licensing implications\n"
            ));
        }

        // Copy the files normally
        for item in &src_files {
            builder.copy_link(
                &builder.src.join(item),
                &plain_dst_src.join(item),
                FileType::Regular,
            );
        }

        // Create the version file
        builder.create(&plain_dst_src.join("version"), &builder.rust_version());

        // Create the files containing git info, to ensure --version outputs the same.
        let write_git_info = |info: Option<&Info>, path: &Path| {
            if let Some(info) = info {
                t!(std::fs::create_dir_all(path));
                channel::write_commit_hash_file(path, &info.sha);
                channel::write_commit_info_file(path, info);
            }
        };
        write_git_info(builder.rust_info().info(), plain_dst_src);
        write_git_info(builder.cargo_info.info(), &plain_dst_src.join("./src/tools/cargo"));

        if builder.config.dist_vendor {
            builder.require_and_update_all_submodules();

            // Vendor packages that are required by opt-dist to collect PGO profiles.
            let pkgs_for_pgo_training = build_helper::LLVM_PGO_CRATES
                .iter()
                .chain(build_helper::RUSTC_PGO_CRATES)
                .map(|pkg| {
                    let mut manifest_path =
                        builder.src.join("./src/tools/rustc-perf/collector/compile-benchmarks");
                    manifest_path.push(pkg);
                    manifest_path.push("Cargo.toml");
                    manifest_path
                });

            // Vendor all Cargo dependencies
            let vendor = builder.ensure(Vendor {
                sync_args: pkgs_for_pgo_training.collect(),
                versioned_dirs: true,
                root_dir: plain_dst_src.into(),
                output_dir: VENDOR_DIR.into(),
            });

            let cargo_config_dir = plain_dst_src.join(".cargo");
            builder.create_dir(&cargo_config_dir);
            builder.create(&cargo_config_dir.join("config.toml"), &vendor.config);
        }

        // Delete extraneous directories
        // FIXME: if we're managed by git, we should probably instead ask git if the given path
        // is managed by it?
        for entry in walkdir::WalkDir::new(tarball.image_dir())
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().is_dir() && entry.path().file_name() == Some(OsStr::new("__pycache__"))
            {
                t!(fs::remove_dir_all(entry.path()));
            }
        }

        tarball.bare()
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Cargo {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Cargo {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = should_build_extended_tool(run.builder, "cargo");
        run.alias("cargo").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Cargo {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(compile::Rustc::new(compiler, target));

        let cargo = builder.ensure(tool::Cargo { compiler, target });
        let src = builder.src.join("src/tools/cargo");
        let etc = src.join("src/etc");

        // Prepare the image directory
        let mut tarball = Tarball::new(builder, "cargo", &target.triple);
        tarball.set_overlay(OverlayKind::Cargo);

        tarball.add_file(&cargo.tool_path, "bin", FileType::Executable);
        tarball.add_file(etc.join("_cargo"), "share/zsh/site-functions", FileType::Regular);
        tarball.add_renamed_file(
            etc.join("cargo.bashcomp.sh"),
            "etc/bash_completion.d",
            "cargo",
            FileType::Regular,
        );
        tarball.add_dir(etc.join("man"), "share/man/man1");
        tarball.add_legal_and_readme_to("share/doc/cargo");

        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct RustAnalyzer {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustAnalyzer {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = should_build_extended_tool(run.builder, "rust-analyzer");
        run.alias("rust-analyzer").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustAnalyzer {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(compile::Rustc::new(compiler, target));

        let rust_analyzer = builder.ensure(tool::RustAnalyzer { compiler, target });

        let mut tarball = Tarball::new(builder, "rust-analyzer", &target.triple);
        tarball.set_overlay(OverlayKind::RustAnalyzer);
        tarball.is_preview(true);
        tarball.add_file(&rust_analyzer.tool_path, "bin", FileType::Executable);
        tarball.add_legal_and_readme_to("share/doc/rust-analyzer");
        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Clippy {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Clippy {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = should_build_extended_tool(run.builder, "clippy");
        run.alias("clippy").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Clippy {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(compile::Rustc::new(compiler, target));

        // Prepare the image directory
        // We expect clippy to build, because we've exited this step above if tool
        // state for clippy isn't testing.
        let clippy = builder.ensure(tool::Clippy { compiler, target });
        let cargoclippy = builder.ensure(tool::CargoClippy { compiler, target });

        let mut tarball = Tarball::new(builder, "clippy", &target.triple);
        tarball.set_overlay(OverlayKind::Clippy);
        tarball.is_preview(true);
        tarball.add_file(&clippy.tool_path, "bin", FileType::Executable);
        tarball.add_file(&cargoclippy.tool_path, "bin", FileType::Executable);
        tarball.add_legal_and_readme_to("share/doc/clippy");
        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Miri {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Miri {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = should_build_extended_tool(run.builder, "miri");
        run.alias("miri").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Miri {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        // This prevents miri from being built for "dist" or "install"
        // on the stable/beta channels. It is a nightly-only tool and should
        // not be included.
        if !builder.build.unstable_features() {
            return None;
        }

        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(compile::Rustc::new(compiler, target));

        let miri = builder.ensure(tool::Miri { compiler, target });
        let cargomiri = builder.ensure(tool::CargoMiri { compiler, target });

        let mut tarball = Tarball::new(builder, "miri", &target.triple);
        tarball.set_overlay(OverlayKind::Miri);
        tarball.is_preview(true);
        tarball.add_file(&miri.tool_path, "bin", FileType::Executable);
        tarball.add_file(&cargomiri.tool_path, "bin", FileType::Executable);
        tarball.add_legal_and_readme_to("share/doc/miri");
        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct CodegenBackend {
    pub compiler: Compiler,
    pub backend: String,
}

impl Step for CodegenBackend {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("compiler/rustc_codegen_cranelift")
    }

    fn make_run(run: RunConfig<'_>) {
        for backend in run.builder.config.codegen_backends(run.target) {
            if backend == "llvm" {
                continue; // Already built as part of rustc
            }

            run.builder.ensure(CodegenBackend {
                compiler: run.builder.compiler(run.builder.top_stage, run.target),
                backend: backend.clone(),
            });
        }
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        if builder.config.dry_run() {
            return None;
        }

        // This prevents rustc_codegen_cranelift from being built for "dist"
        // or "install" on the stable/beta channels. It is not yet stable and
        // should not be included.
        if !builder.build.unstable_features() {
            return None;
        }

        if !builder.config.codegen_backends(self.compiler.host).contains(&self.backend.to_string())
        {
            return None;
        }

        if self.backend == "cranelift" && !target_supports_cranelift_backend(self.compiler.host) {
            builder.info("target not supported by rustc_codegen_cranelift. skipping");
            return None;
        }

        let compiler = self.compiler;
        let backend = self.backend;

        let mut tarball =
            Tarball::new(builder, &format!("rustc-codegen-{backend}"), &compiler.host.triple);
        if backend == "cranelift" {
            tarball.set_overlay(OverlayKind::RustcCodegenCranelift);
        } else {
            panic!("Unknown backend rustc_codegen_{backend}");
        }
        tarball.is_preview(true);
        tarball.add_legal_and_readme_to(format!("share/doc/rustc_codegen_{backend}"));

        let src = builder.sysroot(compiler);
        let backends_src = builder.sysroot_codegen_backends(compiler);
        let backends_rel = backends_src
            .strip_prefix(src)
            .unwrap()
            .strip_prefix(builder.sysroot_libdir_relative(compiler))
            .unwrap();
        // Don't use custom libdir here because ^lib/ will be resolved again with installer
        let backends_dst = PathBuf::from("lib").join(backends_rel);

        let backend_name = format!("rustc_codegen_{backend}");
        let mut found_backend = false;
        for backend in fs::read_dir(&backends_src).unwrap() {
            let file_name = backend.unwrap().file_name();
            if file_name.to_str().unwrap().contains(&backend_name) {
                tarball.add_file(
                    backends_src.join(file_name),
                    &backends_dst,
                    FileType::NativeLibrary,
                );
                found_backend = true;
            }
        }
        assert!(found_backend);

        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Rustfmt {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Rustfmt {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = should_build_extended_tool(run.builder, "rustfmt");
        run.alias("rustfmt").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustfmt {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(compile::Rustc::new(compiler, target));

        let rustfmt = builder.ensure(tool::Rustfmt { compiler, target });
        let cargofmt = builder.ensure(tool::Cargofmt { compiler, target });
        let mut tarball = Tarball::new(builder, "rustfmt", &target.triple);
        tarball.set_overlay(OverlayKind::Rustfmt);
        tarball.is_preview(true);
        tarball.add_file(&rustfmt.tool_path, "bin", FileType::Executable);
        tarball.add_file(&cargofmt.tool_path, "bin", FileType::Executable);
        tarball.add_legal_and_readme_to("share/doc/rustfmt");
        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct Extended {
    stage: u32,
    host: TargetSelection,
    target: TargetSelection,
}

impl Step for Extended {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.alias("extended").default_condition(builder.config.extended)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Extended {
            stage: run.builder.top_stage,
            host: run.builder.config.build,
            target: run.target,
        });
    }

    /// Creates a combined installer for the specified target in the provided stage.
    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let stage = self.stage;
        let compiler = builder.compiler_for(self.stage, self.host, self.target);

        builder.info(&format!("Dist extended stage{} ({})", compiler.stage, target));

        let mut tarballs = Vec::new();
        let mut built_tools = HashSet::new();
        macro_rules! add_component {
            ($name:expr => $step:expr) => {
                if let Some(tarball) = builder.ensure_if_default($step, Kind::Dist) {
                    tarballs.push(tarball);
                    built_tools.insert($name);
                }
            };
        }

        // When rust-std package split from rustc, we needed to ensure that during
        // upgrades rustc was upgraded before rust-std. To avoid rustc clobbering
        // the std files during uninstall. To do this ensure that rustc comes
        // before rust-std in the list below.
        tarballs.push(builder.ensure(Rustc { compiler: builder.compiler(stage, target) }));
        tarballs.push(builder.ensure(Std { compiler, target }).expect("missing std"));

        if target.is_windows_gnu() {
            tarballs.push(builder.ensure(Mingw { host: target }).expect("missing mingw"));
        }

        add_component!("rust-docs" => Docs { host: target });
        add_component!("rust-json-docs" => JsonDocs { host: target });
        add_component!("cargo" => Cargo { compiler, target });
        add_component!("rustfmt" => Rustfmt { compiler, target });
        add_component!("rust-analyzer" => RustAnalyzer { compiler, target });
        add_component!("llvm-components" => LlvmTools { target });
        add_component!("clippy" => Clippy { compiler, target });
        add_component!("miri" => Miri { compiler, target });
        add_component!("analysis" => Analysis { compiler, target });
        add_component!("rustc-codegen-cranelift" => CodegenBackend {
            compiler: builder.compiler(stage, target),
            backend: "cranelift".to_string(),
        });
        add_component!("llvm-bitcode-linker" => LlvmBitcodeLinker {compiler, target});

        let etc = builder.src.join("src/etc/installer");

        // Avoid producing tarballs during a dry run.
        if builder.config.dry_run() {
            return;
        }

        let tarball = Tarball::new(builder, "rust", &target.triple);
        let generated = tarball.combine(&tarballs);

        let tmp = tmpdir(builder).join("combined-tarball");
        let work = generated.work_dir();

        let mut license = String::new();
        license += &builder.read(&builder.src.join("COPYRIGHT"));
        license += &builder.read(&builder.src.join("LICENSE-APACHE"));
        license += &builder.read(&builder.src.join("LICENSE-MIT"));
        license.push('\n');
        license.push('\n');

        let rtf = r"{\rtf1\ansi\deff0{\fonttbl{\f0\fnil\fcharset0 Arial;}}\nowwrap\fs18";
        let mut rtf = rtf.to_string();
        rtf.push('\n');
        for line in license.lines() {
            rtf.push_str(line);
            rtf.push_str("\\line ");
        }
        rtf.push('}');

        fn filter(contents: &str, marker: &str) -> String {
            let start = format!("tool-{marker}-start");
            let end = format!("tool-{marker}-end");
            let mut lines = Vec::new();
            let mut omitted = false;
            for line in contents.lines() {
                if line.contains(&start) {
                    omitted = true;
                } else if line.contains(&end) {
                    omitted = false;
                } else if !omitted {
                    lines.push(line);
                }
            }

            lines.join("\n")
        }

        let xform = |p: &Path| {
            let mut contents = t!(fs::read_to_string(p));
            for tool in &["miri", "rust-docs"] {
                if !built_tools.contains(tool) {
                    contents = filter(&contents, tool);
                }
            }
            let ret = tmp.join(p.file_name().unwrap());
            t!(fs::write(&ret, &contents));
            ret
        };

        if target.contains("apple-darwin") {
            builder.info("building pkg installer");
            let pkg = tmp.join("pkg");
            let _ = fs::remove_dir_all(&pkg);

            let pkgbuild = |component: &str| {
                let mut cmd = command("pkgbuild");
                cmd.arg("--identifier")
                    .arg(format!("org.rust-lang.{component}"))
                    .arg("--scripts")
                    .arg(pkg.join(component))
                    .arg("--nopayload")
                    .arg(pkg.join(component).with_extension("pkg"));
                cmd.run(builder);
            };

            let prepare = |name: &str| {
                builder.create_dir(&pkg.join(name));
                builder.cp_link_r(
                    &work.join(format!("{}-{}", pkgname(builder, name), target.triple)),
                    &pkg.join(name),
                );
                builder.install(&etc.join("pkg/postinstall"), &pkg.join(name), FileType::Script);
                pkgbuild(name);
            };
            prepare("rustc");
            prepare("cargo");
            prepare("rust-std");
            prepare("rust-analysis");

            for tool in &[
                "clippy",
                "rustfmt",
                "rust-analyzer",
                "rust-docs",
                "miri",
                "rustc-codegen-cranelift",
            ] {
                if built_tools.contains(tool) {
                    prepare(tool);
                }
            }
            // create an 'uninstall' package
            builder.install(&etc.join("pkg/postinstall"), &pkg.join("uninstall"), FileType::Script);
            pkgbuild("uninstall");

            builder.create_dir(&pkg.join("res"));
            builder.create(&pkg.join("res/LICENSE.txt"), &license);
            builder.install(&etc.join("gfx/rust-logo.png"), &pkg.join("res"), FileType::Regular);
            let mut cmd = command("productbuild");
            cmd.arg("--distribution")
                .arg(xform(&etc.join("pkg/Distribution.xml")))
                .arg("--resources")
                .arg(pkg.join("res"))
                .arg(distdir(builder).join(format!(
                    "{}-{}.pkg",
                    pkgname(builder, "rust"),
                    target.triple
                )))
                .arg("--package-path")
                .arg(&pkg);
            let _time = timeit(builder);
            cmd.run(builder);
        }

        // FIXME(mati865): `gnullvm` here is temporary, remove it once it can host itself
        if target.is_windows() && !target.contains("gnullvm") {
            let exe = tmp.join("exe");
            let _ = fs::remove_dir_all(&exe);

            let prepare = |name: &str| {
                builder.create_dir(&exe.join(name));
                let dir = if name == "rust-std" || name == "rust-analysis" {
                    format!("{}-{}", name, target.triple)
                } else if name == "rust-analyzer" {
                    "rust-analyzer-preview".to_string()
                } else if name == "clippy" {
                    "clippy-preview".to_string()
                } else if name == "rustfmt" {
                    "rustfmt-preview".to_string()
                } else if name == "miri" {
                    "miri-preview".to_string()
                } else if name == "rustc-codegen-cranelift" {
                    // FIXME add installer support for cg_clif once it is ready to be distributed on
                    // windows.
                    unreachable!("cg_clif shouldn't be built for windows");
                } else {
                    name.to_string()
                };
                builder.cp_link_r(
                    &work.join(format!("{}-{}", pkgname(builder, name), target.triple)).join(dir),
                    &exe.join(name),
                );
                builder.remove(&exe.join(name).join("manifest.in"));
            };
            prepare("rustc");
            prepare("cargo");
            prepare("rust-analysis");
            prepare("rust-std");
            for tool in &["clippy", "rustfmt", "rust-analyzer", "rust-docs", "miri"] {
                if built_tools.contains(tool) {
                    prepare(tool);
                }
            }
            if target.is_windows_gnu() {
                prepare("rust-mingw");
            }

            builder.install(&etc.join("gfx/rust-logo.ico"), &exe, FileType::Regular);

            // Generate msi installer
            let wix_path = env::var_os("WIX")
                .expect("`WIX` environment variable must be set for generating MSI installer(s).");
            let wix = PathBuf::from(wix_path);
            let heat = wix.join("bin/heat.exe");
            let candle = wix.join("bin/candle.exe");
            let light = wix.join("bin/light.exe");

            let heat_flags = ["-nologo", "-gg", "-sfrag", "-srd", "-sreg"];
            command(&heat)
                .current_dir(&exe)
                .arg("dir")
                .arg("rustc")
                .args(heat_flags)
                .arg("-cg")
                .arg("RustcGroup")
                .arg("-dr")
                .arg("Rustc")
                .arg("-var")
                .arg("var.RustcDir")
                .arg("-out")
                .arg(exe.join("RustcGroup.wxs"))
                .run(builder);
            if built_tools.contains("rust-docs") {
                command(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("rust-docs")
                    .args(heat_flags)
                    .arg("-cg")
                    .arg("DocsGroup")
                    .arg("-dr")
                    .arg("Docs")
                    .arg("-var")
                    .arg("var.DocsDir")
                    .arg("-out")
                    .arg(exe.join("DocsGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/squash-components.xsl"))
                    .run(builder);
            }
            command(&heat)
                .current_dir(&exe)
                .arg("dir")
                .arg("cargo")
                .args(heat_flags)
                .arg("-cg")
                .arg("CargoGroup")
                .arg("-dr")
                .arg("Cargo")
                .arg("-var")
                .arg("var.CargoDir")
                .arg("-out")
                .arg(exe.join("CargoGroup.wxs"))
                .arg("-t")
                .arg(etc.join("msi/remove-duplicates.xsl"))
                .run(builder);
            command(&heat)
                .current_dir(&exe)
                .arg("dir")
                .arg("rust-std")
                .args(heat_flags)
                .arg("-cg")
                .arg("StdGroup")
                .arg("-dr")
                .arg("Std")
                .arg("-var")
                .arg("var.StdDir")
                .arg("-out")
                .arg(exe.join("StdGroup.wxs"))
                .run(builder);
            if built_tools.contains("rust-analyzer") {
                command(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("rust-analyzer")
                    .args(heat_flags)
                    .arg("-cg")
                    .arg("RustAnalyzerGroup")
                    .arg("-dr")
                    .arg("RustAnalyzer")
                    .arg("-var")
                    .arg("var.RustAnalyzerDir")
                    .arg("-out")
                    .arg(exe.join("RustAnalyzerGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/remove-duplicates.xsl"))
                    .run(builder);
            }
            if built_tools.contains("clippy") {
                command(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("clippy")
                    .args(heat_flags)
                    .arg("-cg")
                    .arg("ClippyGroup")
                    .arg("-dr")
                    .arg("Clippy")
                    .arg("-var")
                    .arg("var.ClippyDir")
                    .arg("-out")
                    .arg(exe.join("ClippyGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/remove-duplicates.xsl"))
                    .run(builder);
            }
            if built_tools.contains("rustfmt") {
                command(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("rustfmt")
                    .args(heat_flags)
                    .arg("-cg")
                    .arg("RustFmtGroup")
                    .arg("-dr")
                    .arg("RustFmt")
                    .arg("-var")
                    .arg("var.RustFmtDir")
                    .arg("-out")
                    .arg(exe.join("RustFmtGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/remove-duplicates.xsl"))
                    .run(builder);
            }
            if built_tools.contains("miri") {
                command(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("miri")
                    .args(heat_flags)
                    .arg("-cg")
                    .arg("MiriGroup")
                    .arg("-dr")
                    .arg("Miri")
                    .arg("-var")
                    .arg("var.MiriDir")
                    .arg("-out")
                    .arg(exe.join("MiriGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/remove-duplicates.xsl"))
                    .run(builder);
            }
            command(&heat)
                .current_dir(&exe)
                .arg("dir")
                .arg("rust-analysis")
                .args(heat_flags)
                .arg("-cg")
                .arg("AnalysisGroup")
                .arg("-dr")
                .arg("Analysis")
                .arg("-var")
                .arg("var.AnalysisDir")
                .arg("-out")
                .arg(exe.join("AnalysisGroup.wxs"))
                .arg("-t")
                .arg(etc.join("msi/remove-duplicates.xsl"))
                .run(builder);
            if target.is_windows_gnu() {
                command(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("rust-mingw")
                    .args(heat_flags)
                    .arg("-cg")
                    .arg("GccGroup")
                    .arg("-dr")
                    .arg("Gcc")
                    .arg("-var")
                    .arg("var.GccDir")
                    .arg("-out")
                    .arg(exe.join("GccGroup.wxs"))
                    .run(builder);
            }

            let candle = |input: &Path| {
                let output = exe.join(input.file_stem().unwrap()).with_extension("wixobj");
                let arch = if target.contains("x86_64") { "x64" } else { "x86" };
                let mut cmd = command(&candle);
                cmd.current_dir(&exe)
                    .arg("-nologo")
                    .arg("-dRustcDir=rustc")
                    .arg("-dCargoDir=cargo")
                    .arg("-dStdDir=rust-std")
                    .arg("-dAnalysisDir=rust-analysis")
                    .arg("-arch")
                    .arg(arch)
                    .arg("-out")
                    .arg(&output)
                    .arg(input);
                add_env(builder, &mut cmd, target, &built_tools);

                if built_tools.contains("clippy") {
                    cmd.arg("-dClippyDir=clippy");
                }
                if built_tools.contains("rustfmt") {
                    cmd.arg("-dRustFmtDir=rustfmt");
                }
                if built_tools.contains("rust-docs") {
                    cmd.arg("-dDocsDir=rust-docs");
                }
                if built_tools.contains("rust-analyzer") {
                    cmd.arg("-dRustAnalyzerDir=rust-analyzer");
                }
                if built_tools.contains("miri") {
                    cmd.arg("-dMiriDir=miri");
                }
                if target.is_windows_gnu() {
                    cmd.arg("-dGccDir=rust-mingw");
                }
                cmd.run(builder);
            };
            candle(&xform(&etc.join("msi/rust.wxs")));
            candle(&etc.join("msi/ui.wxs"));
            candle(&etc.join("msi/rustwelcomedlg.wxs"));
            candle("RustcGroup.wxs".as_ref());
            if built_tools.contains("rust-docs") {
                candle("DocsGroup.wxs".as_ref());
            }
            candle("CargoGroup.wxs".as_ref());
            candle("StdGroup.wxs".as_ref());
            if built_tools.contains("clippy") {
                candle("ClippyGroup.wxs".as_ref());
            }
            if built_tools.contains("rustfmt") {
                candle("RustFmtGroup.wxs".as_ref());
            }
            if built_tools.contains("miri") {
                candle("MiriGroup.wxs".as_ref());
            }
            if built_tools.contains("rust-analyzer") {
                candle("RustAnalyzerGroup.wxs".as_ref());
            }
            candle("AnalysisGroup.wxs".as_ref());

            if target.is_windows_gnu() {
                candle("GccGroup.wxs".as_ref());
            }

            builder.create(&exe.join("LICENSE.rtf"), &rtf);
            builder.install(&etc.join("gfx/banner.bmp"), &exe, FileType::Regular);
            builder.install(&etc.join("gfx/dialogbg.bmp"), &exe, FileType::Regular);

            builder.info(&format!("building `msi` installer with {light:?}"));
            let filename = format!("{}-{}.msi", pkgname(builder, "rust"), target.triple);
            let mut cmd = command(&light);
            cmd.arg("-nologo")
                .arg("-ext")
                .arg("WixUIExtension")
                .arg("-ext")
                .arg("WixUtilExtension")
                .arg("-out")
                .arg(exe.join(&filename))
                .arg("rust.wixobj")
                .arg("ui.wixobj")
                .arg("rustwelcomedlg.wixobj")
                .arg("RustcGroup.wixobj")
                .arg("CargoGroup.wixobj")
                .arg("StdGroup.wixobj")
                .arg("AnalysisGroup.wixobj")
                .current_dir(&exe);

            if built_tools.contains("clippy") {
                cmd.arg("ClippyGroup.wixobj");
            }
            if built_tools.contains("rustfmt") {
                cmd.arg("RustFmtGroup.wixobj");
            }
            if built_tools.contains("miri") {
                cmd.arg("MiriGroup.wixobj");
            }
            if built_tools.contains("rust-analyzer") {
                cmd.arg("RustAnalyzerGroup.wixobj");
            }
            if built_tools.contains("rust-docs") {
                cmd.arg("DocsGroup.wixobj");
            }

            if target.is_windows_gnu() {
                cmd.arg("GccGroup.wixobj");
            }
            // ICE57 wrongly complains about the shortcuts
            cmd.arg("-sice:ICE57");

            let _time = timeit(builder);
            cmd.run(builder);

            if !builder.config.dry_run() {
                t!(move_file(exe.join(&filename), distdir(builder).join(&filename)));
            }
        }
    }
}

fn add_env(
    builder: &Builder<'_>,
    cmd: &mut BootstrapCommand,
    target: TargetSelection,
    built_tools: &HashSet<&'static str>,
) {
    let mut parts = builder.version.split('.');
    cmd.env("CFG_RELEASE_INFO", builder.rust_version())
        .env("CFG_RELEASE_NUM", &builder.version)
        .env("CFG_RELEASE", builder.rust_release())
        .env("CFG_VER_MAJOR", parts.next().unwrap())
        .env("CFG_VER_MINOR", parts.next().unwrap())
        .env("CFG_VER_PATCH", parts.next().unwrap())
        .env("CFG_VER_BUILD", "0") // just needed to build
        .env("CFG_PACKAGE_VERS", builder.rust_package_vers())
        .env("CFG_PACKAGE_NAME", pkgname(builder, "rust"))
        .env("CFG_BUILD", target.triple)
        .env("CFG_CHANNEL", &builder.config.channel);

    if target.contains("windows-gnullvm") {
        cmd.env("CFG_MINGW", "1").env("CFG_ABI", "LLVM");
    } else if target.is_windows_gnu() {
        cmd.env("CFG_MINGW", "1").env("CFG_ABI", "GNU");
    } else {
        cmd.env("CFG_MINGW", "0").env("CFG_ABI", "MSVC");
    }

    // ensure these variables are defined
    let mut define_optional_tool = |tool_name: &str, env_name: &str| {
        cmd.env(env_name, if built_tools.contains(tool_name) { "1" } else { "0" });
    };
    define_optional_tool("rustfmt", "CFG_RUSTFMT");
    define_optional_tool("clippy", "CFG_CLIPPY");
    define_optional_tool("miri", "CFG_MIRI");
    define_optional_tool("rust-analyzer", "CFG_RA");
}

fn install_llvm_file(
    builder: &Builder<'_>,
    source: &Path,
    destination: &Path,
    install_symlink: bool,
) {
    if builder.config.dry_run() {
        return;
    }

    if source.is_symlink() {
        // If we have a symlink like libLLVM-18.so -> libLLVM.so.18.1, install the target of the
        // symlink, which is what will actually get loaded at runtime.
        builder.install(&t!(fs::canonicalize(source)), destination, FileType::NativeLibrary);

        let full_dest = destination.join(source.file_name().unwrap());
        if install_symlink {
            // For download-ci-llvm, also install the symlink, to match what LLVM does. Using a
            // symlink is fine here, as this is not a rustup component.
            builder.copy_link(source, &full_dest, FileType::NativeLibrary);
        } else {
            // Otherwise, replace the symlink with an equivalent linker script. This is used when
            // projects like miri link against librustc_driver.so. We don't use a symlink, as
            // these are not allowed inside rustup components.
            let link = t!(fs::read_link(source));
            let mut linker_script = t!(fs::File::create(full_dest));
            t!(write!(linker_script, "INPUT({})\n", link.display()));

            // We also want the linker script to have the same mtime as the source, otherwise it
            // can trigger rebuilds.
            let meta = t!(fs::metadata(source));
            if let Ok(mtime) = meta.modified() {
                t!(linker_script.set_modified(mtime));
            }
        }
    } else {
        builder.install(source, destination, FileType::NativeLibrary);
    }
}

/// Maybe add LLVM object files to the given destination lib-dir. Allows either static or dynamic linking.
///
/// Returns whether the files were actually copied.
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "trace",
        name = "maybe_install_llvm",
        skip_all,
        fields(target = ?target, dst_libdir = ?dst_libdir, install_symlink = install_symlink),
    ),
)]
fn maybe_install_llvm(
    builder: &Builder<'_>,
    target: TargetSelection,
    dst_libdir: &Path,
    install_symlink: bool,
) -> bool {
    // If the LLVM was externally provided, then we don't currently copy
    // artifacts into the sysroot. This is not necessarily the right
    // choice (in particular, it will require the LLVM dylib to be in
    // the linker's load path at runtime), but the common use case for
    // external LLVMs is distribution provided LLVMs, and in that case
    // they're usually in the standard search path (e.g., /usr/lib) and
    // copying them here is going to cause problems as we may end up
    // with the wrong files and isn't what distributions want.
    //
    // This behavior may be revisited in the future though.
    //
    // NOTE: this intentionally doesn't use `is_rust_llvm`; whether this is patched or not doesn't matter,
    // we only care if the shared object itself is managed by bootstrap.
    //
    // If the LLVM is coming from ourselves (just from CI) though, we
    // still want to install it, as it otherwise won't be available.
    if builder.config.is_system_llvm(target) {
        trace!("system LLVM requested, no install");
        return false;
    }

    // On macOS, rustc (and LLVM tools) link to an unversioned libLLVM.dylib
    // instead of libLLVM-11-rust-....dylib, as on linux. It's not entirely
    // clear why this is the case, though. llvm-config will emit the versioned
    // paths and we don't want those in the sysroot (as we're expecting
    // unversioned paths).
    if target.contains("apple-darwin") && builder.llvm_link_shared() {
        let src_libdir = builder.llvm_out(target).join("lib");
        let llvm_dylib_path = src_libdir.join("libLLVM.dylib");
        if llvm_dylib_path.exists() {
            builder.install(&llvm_dylib_path, dst_libdir, FileType::NativeLibrary);
        }
        !builder.config.dry_run()
    } else if let llvm::LlvmBuildStatus::AlreadyBuilt(llvm::LlvmResult { llvm_config, .. }) =
        llvm::prebuilt_llvm_config(builder, target, true)
    {
        trace!("LLVM already built, installing LLVM files");
        let mut cmd = command(llvm_config);
        cmd.arg("--libfiles");
        builder.verbose(|| println!("running {cmd:?}"));
        let files = cmd.run_capture_stdout(builder).stdout();
        let build_llvm_out = &builder.llvm_out(builder.config.build);
        let target_llvm_out = &builder.llvm_out(target);
        for file in files.trim_end().split(' ') {
            // If we're not using a custom LLVM, make sure we package for the target.
            let file = if let Ok(relative_path) = Path::new(file).strip_prefix(build_llvm_out) {
                target_llvm_out.join(relative_path)
            } else {
                PathBuf::from(file)
            };
            install_llvm_file(builder, &file, dst_libdir, install_symlink);
        }
        !builder.config.dry_run()
    } else {
        false
    }
}

/// Maybe add libLLVM.so to the target lib-dir for linking.
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "trace",
        name = "maybe_install_llvm_target",
        skip_all,
        fields(
            llvm_link_shared = ?builder.llvm_link_shared(),
            target = ?target,
            sysroot = ?sysroot,
        ),
    ),
)]
pub fn maybe_install_llvm_target(builder: &Builder<'_>, target: TargetSelection, sysroot: &Path) {
    let dst_libdir = sysroot.join("lib/rustlib").join(target).join("lib");
    // We do not need to copy LLVM files into the sysroot if it is not
    // dynamically linked; it is already included into librustc_llvm
    // statically.
    if builder.llvm_link_shared() {
        maybe_install_llvm(builder, target, &dst_libdir, false);
    }
}

/// Maybe add libLLVM.so to the runtime lib-dir for rustc itself.
#[cfg_attr(
    feature = "tracing",
    instrument(
        level = "trace",
        name = "maybe_install_llvm_runtime",
        skip_all,
        fields(
            llvm_link_shared = ?builder.llvm_link_shared(),
            target = ?target,
            sysroot = ?sysroot,
        ),
    ),
)]
pub fn maybe_install_llvm_runtime(builder: &Builder<'_>, target: TargetSelection, sysroot: &Path) {
    let dst_libdir = sysroot.join(builder.sysroot_libdir_relative(Compiler::new(1, target)));
    // We do not need to copy LLVM files into the sysroot if it is not
    // dynamically linked; it is already included into librustc_llvm
    // statically.
    if builder.llvm_link_shared() {
        maybe_install_llvm(builder, target, &dst_libdir, false);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LlvmTools {
    pub target: TargetSelection,
}

impl Step for LlvmTools {
    type Output = Option<GeneratedTarball>;
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = should_build_extended_tool(run.builder, "llvm-tools");

        let mut run = run.alias("llvm-tools");
        for tool in LLVM_TOOLS {
            run = run.alias(tool);
        }

        run.default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(LlvmTools { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        fn tools_to_install(paths: &[PathBuf]) -> Vec<&'static str> {
            let mut tools = vec![];

            for path in paths {
                let path = path.to_str().unwrap();

                // Include all tools if path is 'llvm-tools'.
                if path == "llvm-tools" {
                    return LLVM_TOOLS.to_owned();
                }

                for tool in LLVM_TOOLS {
                    if path == *tool {
                        tools.push(*tool);
                    }
                }
            }

            // If no specific tool is requested, include all tools.
            if tools.is_empty() {
                tools = LLVM_TOOLS.to_owned();
            }

            tools
        }

        let target = self.target;

        // Run only if a custom llvm-config is not used
        if let Some(config) = builder.config.target_config.get(&target)
            && !builder.config.llvm_from_ci
            && config.llvm_config.is_some()
        {
            builder.info(&format!("Skipping LlvmTools ({target}): external LLVM"));
            return None;
        }

        if !builder.config.dry_run() {
            builder.require_submodule("src/llvm-project", None);
        }

        builder.ensure(crate::core::build_steps::llvm::Llvm { target });

        let mut tarball = Tarball::new(builder, "llvm-tools", &target.triple);
        tarball.set_overlay(OverlayKind::Llvm);
        tarball.is_preview(true);

        if builder.config.llvm_tools_enabled {
            // Prepare the image directory
            let src_bindir = builder.llvm_out(target).join("bin");
            let dst_bindir = format!("lib/rustlib/{}/bin", target.triple);
            for tool in tools_to_install(&builder.paths) {
                let exe = src_bindir.join(exe(tool, target));
                // When using `download-ci-llvm`, some of the tools may not exist, so skip trying to copy them.
                if !exe.exists() && builder.config.llvm_from_ci {
                    eprintln!("{} does not exist; skipping copy", exe.display());
                    continue;
                }

                tarball.add_file(&exe, &dst_bindir, FileType::Executable);
            }
        }

        // Copy libLLVM.so to the target lib dir as well, so the RPATH like
        // `$ORIGIN/../lib` can find it. It may also be used as a dependency
        // of `rustc-dev` to support the inherited `-lLLVM` when using the
        // compiler libraries.
        maybe_install_llvm_target(builder, target, tarball.image_dir());

        Some(tarball.generate())
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, Hash, PartialEq, Eq)]
pub struct LlvmBitcodeLinker {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for LlvmBitcodeLinker {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let default = should_build_extended_tool(run.builder, "llvm-bitcode-linker");
        run.alias("llvm-bitcode-linker").default_condition(default)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(LlvmBitcodeLinker {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(compile::Rustc::new(compiler, target));

        let llbc_linker =
            builder.ensure(tool::LlvmBitcodeLinker { compiler, target, extra_features: vec![] });

        let self_contained_bin_dir = format!("lib/rustlib/{}/bin/self-contained", target.triple);

        // Prepare the image directory
        let mut tarball = Tarball::new(builder, "llvm-bitcode-linker", &target.triple);
        tarball.set_overlay(OverlayKind::LlvmBitcodeLinker);
        tarball.is_preview(true);

        tarball.add_file(&llbc_linker.tool_path, self_contained_bin_dir, FileType::Executable);

        Some(tarball.generate())
    }
}

/// Tarball intended for internal consumption to ease rustc/std development.
///
/// Should not be considered stable by end users.
///
/// In practice, this is the tarball that gets downloaded and used by
/// `llvm.download-ci-llvm`.
///
/// (Don't confuse this with [`RustcDev`], with a `c`!)
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RustDev {
    pub target: TargetSelection,
}

impl Step for RustDev {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("rust-dev")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustDev { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let target = self.target;

        /* run only if llvm-config isn't used */
        if let Some(config) = builder.config.target_config.get(&target)
            && let Some(ref _s) = config.llvm_config
        {
            builder.info(&format!("Skipping RustDev ({target}): external LLVM"));
            return None;
        }

        if !builder.config.dry_run() {
            builder.require_submodule("src/llvm-project", None);
        }

        let mut tarball = Tarball::new(builder, "rust-dev", &target.triple);
        tarball.set_overlay(OverlayKind::Llvm);
        // LLVM requires a shared object symlink to exist on some platforms.
        tarball.permit_symlinks(true);

        builder.ensure(crate::core::build_steps::llvm::Llvm { target });

        let src_bindir = builder.llvm_out(target).join("bin");
        // If updating this, you likely want to change
        // src/bootstrap/download-ci-llvm-stamp as well, otherwise local users
        // will not pick up the extra file until LLVM gets bumped.
        // We should include all the build artifacts obtained from a source build,
        // so that you can use the downloadable LLVM as if you’ve just run a full source build.
        if src_bindir.exists() {
            for entry in walkdir::WalkDir::new(&src_bindir) {
                let entry = t!(entry);
                if entry.file_type().is_file() && !entry.path_is_symlink() {
                    let name = entry.file_name().to_str().unwrap();
                    tarball.add_file(src_bindir.join(name), "bin", FileType::Executable);
                }
            }
        }

        if builder.config.lld_enabled {
            // We want to package `lld` to use it with `download-ci-llvm`.
            let lld_out = builder.ensure(crate::core::build_steps::llvm::Lld { target });

            // We don't build LLD on some platforms, so only add it if it exists
            let lld_path = lld_out.join("bin").join(exe("lld", target));
            if lld_path.exists() {
                tarball.add_file(&lld_path, "bin", FileType::Executable);
            }
        }

        tarball.add_file(builder.llvm_filecheck(target), "bin", FileType::Executable);

        // Copy the include directory as well; needed mostly to build
        // librustc_llvm properly (e.g., llvm-config.h is in here). But also
        // just broadly useful to be able to link against the bundled LLVM.
        tarball.add_dir(builder.llvm_out(target).join("include"), "include");

        // Copy libLLVM.so to the target lib dir as well, so the RPATH like
        // `$ORIGIN/../lib` can find it. It may also be used as a dependency
        // of `rustc-dev` to support the inherited `-lLLVM` when using the
        // compiler libraries.
        let dst_libdir = tarball.image_dir().join("lib");
        maybe_install_llvm(builder, target, &dst_libdir, true);
        let link_type = if builder.llvm_link_shared() { "dynamic" } else { "static" };
        t!(std::fs::write(tarball.image_dir().join("link-type.txt"), link_type), dst_libdir);

        // Copy the `compiler-rt` source, so that `library/profiler_builtins`
        // can potentially use it to build the profiler runtime without needing
        // to check out the LLVM submodule.
        copy_src_dirs(
            builder,
            &builder.src.join("src").join("llvm-project"),
            &["compiler-rt"],
            // The test subdirectory is much larger than the rest of the source,
            // and we currently don't use these test files anyway.
            &["compiler-rt/test"],
            tarball.image_dir(),
        );

        Some(tarball.generate())
    }
}

/// Tarball intended for internal consumption to ease rustc/std development.
///
/// Should not be considered stable by end users.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Bootstrap {
    pub target: TargetSelection,
}

impl Step for Bootstrap {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = false;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("bootstrap")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Bootstrap { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> Option<GeneratedTarball> {
        let target = self.target;

        let tarball = Tarball::new(builder, "bootstrap", &target.triple);

        let bootstrap_outdir = &builder.bootstrap_out;
        for file in &["bootstrap", "rustc", "rustdoc"] {
            tarball.add_file(
                bootstrap_outdir.join(exe(file, target)),
                "bootstrap/bin",
                FileType::Executable,
            );
        }

        Some(tarball.generate())
    }
}

/// Tarball containing a prebuilt version of the build-manifest tool, intended to be used by the
/// release process to avoid cloning the monorepo and building stuff.
///
/// Should not be considered stable by end users.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BuildManifest {
    pub target: TargetSelection,
}

impl Step for BuildManifest {
    type Output = GeneratedTarball;
    const DEFAULT: bool = false;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("build-manifest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(BuildManifest { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> GeneratedTarball {
        let build_manifest = builder.tool_exe(Tool::BuildManifest);

        let tarball = Tarball::new(builder, "build-manifest", &self.target.triple);
        tarball.add_file(&build_manifest, "bin", FileType::Executable);
        tarball.generate()
    }
}

/// Tarball containing artifacts necessary to reproduce the build of rustc.
///
/// Currently this is the PGO profile data.
///
/// Should not be considered stable by end users.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReproducibleArtifacts {
    pub target: TargetSelection,
}

impl Step for ReproducibleArtifacts {
    type Output = Option<GeneratedTarball>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("reproducible-artifacts")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(ReproducibleArtifacts { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let mut added_anything = false;
        let tarball = Tarball::new(builder, "reproducible-artifacts", &self.target.triple);
        if let Some(path) = builder.config.rust_profile_use.as_ref() {
            tarball.add_file(path, ".", FileType::Regular);
            added_anything = true;
        }
        if let Some(path) = builder.config.llvm_profile_use.as_ref() {
            tarball.add_file(path, ".", FileType::Regular);
            added_anything = true;
        }
        for profile in &builder.config.reproducible_artifacts {
            tarball.add_file(profile, ".", FileType::Regular);
            added_anything = true;
        }
        if added_anything { Some(tarball.generate()) } else { None }
    }
}

/// Tarball containing a prebuilt version of the libgccjit library,
/// needed as a dependency for the GCC codegen backend (similarly to the LLVM
/// backend needing a prebuilt libLLVM).
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Gcc {
    pub target: TargetSelection,
}

impl Step for Gcc {
    type Output = GeneratedTarball;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("gcc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Gcc { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let tarball = Tarball::new(builder, "gcc", &self.target.triple);
        let output = builder.ensure(super::gcc::Gcc { target: self.target });
        tarball.add_file(&output.libgccjit, "lib", FileType::NativeLibrary);
        tarball.generate()
    }
}
