//! Implementation of the various distribution aspects of the compiler.
//!
//! This module is responsible for creating tarballs of the standard library,
//! compiler, and documentation. This ends up being what we distribute to
//! everyone as well.
//!
//! No tarball is actually created literally in this file, but rather we shell
//! out to `rust-installer` still. This may one day be replaced with bits and
//! pieces of `rustup.rs`!

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use build_helper::{output, t};

use crate::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::cache::{Interned, INTERNER};
use crate::compile;
use crate::config::TargetSelection;
use crate::tool::{self, Tool};
use crate::util::{exe, is_dylib, timeit};
use crate::{Compiler, DependencyType, Mode, LLVM_TOOLS};
use time::{self, Timespec};

pub fn pkgname(builder: &Builder<'_>, component: &str) -> String {
    format!("{}-{}", component, builder.rust_package_vers())
}

pub(crate) fn distdir(builder: &Builder<'_>) -> PathBuf {
    builder.out.join("dist")
}

pub fn tmpdir(builder: &Builder<'_>) -> PathBuf {
    builder.out.join("tmp/dist")
}

fn rust_installer(builder: &Builder<'_>) -> Command {
    builder.tool_cmd(Tool::RustInstaller)
}

fn missing_tool(tool_name: &str, skip: bool) {
    if skip {
        println!("Unable to build {}, skipping dist", tool_name)
    } else {
        panic!("Unable to build {}", tool_name)
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Docs {
    pub host: TargetSelection,
}

impl Step for Docs {
    type Output = PathBuf;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/doc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Docs { host: run.target });
    }

    /// Builds the `rust-docs` installer component.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let host = self.host;

        let name = pkgname(builder, "rust-docs");

        if !builder.config.docs {
            return distdir(builder).join(format!("{}-{}.tar.gz", name, host.triple));
        }

        builder.default_doc(None);

        builder.info(&format!("Dist docs ({})", host));
        let _time = timeit(builder);

        let image = tmpdir(builder).join(format!("{}-{}-image", name, host.triple));
        let _ = fs::remove_dir_all(&image);

        let dst = image.join("share/doc/rust/html");
        t!(fs::create_dir_all(&dst));
        let src = builder.doc_out(host);
        builder.cp_r(&src, &dst);
        builder.install(&builder.src.join("src/doc/robots.txt"), &dst, 0o644);

        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust-Documentation")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=Rust-documentation-is-installed.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg(format!("--package-name={}-{}", name, host.triple))
            .arg("--component-name=rust-docs")
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--bulk-dirs=share/doc/rust/html");
        builder.run(&mut cmd);
        builder.remove_dir(&image);

        distdir(builder).join(format!("{}-{}.tar.gz", name, host.triple))
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustcDocs {
    pub host: TargetSelection,
}

impl Step for RustcDocs {
    type Output = PathBuf;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/librustc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustcDocs { host: run.target });
    }

    /// Builds the `rustc-docs` installer component.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let host = self.host;

        let name = pkgname(builder, "rustc-docs");

        if !builder.config.compiler_docs {
            return distdir(builder).join(format!("{}-{}.tar.gz", name, host.triple));
        }

        builder.default_doc(None);

        let image = tmpdir(builder).join(format!("{}-{}-image", name, host.triple));
        let _ = fs::remove_dir_all(&image);

        let dst = image.join("share/doc/rust/html/rustc");
        t!(fs::create_dir_all(&dst));
        let src = builder.compiler_doc_out(host);
        builder.cp_r(&src, &dst);

        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rustc-Documentation")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=Rustc-documentation-is-installed.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg(format!("--package-name={}-{}", name, host.triple))
            .arg("--component-name=rustc-docs")
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--bulk-dirs=share/doc/rust/html/rustc");

        builder.info(&format!("Dist compiler docs ({})", host));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        builder.remove_dir(&image);

        distdir(builder).join(format!("{}-{}.tar.gz", name, host.triple))
    }
}

fn find_files(files: &[&str], path: &[PathBuf]) -> Vec<PathBuf> {
    let mut found = Vec::with_capacity(files.len());

    for file in files {
        let file_path = path.iter().map(|dir| dir.join(file)).find(|p| p.exists());

        if let Some(file_path) = file_path {
            found.push(file_path);
        } else {
            panic!("Could not find '{}' in {:?}", file, path);
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
    //Ask gcc where it keeps its stuff
    let mut cmd = Command::new(builder.cc(target));
    cmd.arg("-print-search-dirs");
    let gcc_out = output(&mut cmd);

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
        //Windows import libs
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
    let dist_bin_dir = rust_root.join("bin/");
    fs::create_dir_all(&dist_bin_dir).expect("creating dist_bin_dir failed");
    for src in rustc_dlls {
        builder.copy_to_folder(&src, &dist_bin_dir);
    }

    //Copy platform tools to platform-specific bin directory
    let target_bin_dir = plat_root
        .join("lib")
        .join("rustlib")
        .join(target.triple)
        .join("bin")
        .join("self-contained");
    fs::create_dir_all(&target_bin_dir).expect("creating target_bin_dir failed");
    for src in target_tools {
        builder.copy_to_folder(&src, &target_bin_dir);
    }

    // Warn windows-gnu users that the bundled GCC cannot compile C files
    builder.create(
        &target_bin_dir.join("GCC-WARNING.txt"),
        "gcc.exe contained in this folder cannot be used for compiling C files - it is only \
         used as a linker. In order to be able to compile projects containing C code use \
         the GCC provided by MinGW or Cygwin.",
    );

    //Copy platform libs to platform-specific lib directory
    let target_lib_dir = plat_root
        .join("lib")
        .join("rustlib")
        .join(target.triple)
        .join("lib")
        .join("self-contained");
    fs::create_dir_all(&target_lib_dir).expect("creating target_lib_dir failed");
    for src in target_libs {
        builder.copy_to_folder(&src, &target_lib_dir);
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Mingw {
    pub host: TargetSelection,
}

impl Step for Mingw {
    type Output = Option<PathBuf>;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Mingw { host: run.target });
    }

    /// Builds the `rust-mingw` installer component.
    ///
    /// This contains all the bits and pieces to run the MinGW Windows targets
    /// without any extra installed software (e.g., we bundle gcc, libraries, etc).
    fn run(self, builder: &Builder<'_>) -> Option<PathBuf> {
        let host = self.host;

        if !host.contains("pc-windows-gnu") {
            return None;
        }

        builder.info(&format!("Dist mingw ({})", host));
        let _time = timeit(builder);
        let name = pkgname(builder, "rust-mingw");
        let image = tmpdir(builder).join(format!("{}-{}-image", name, host.triple));
        let _ = fs::remove_dir_all(&image);
        t!(fs::create_dir_all(&image));

        // The first argument is a "temporary directory" which is just
        // thrown away (this contains the runtime DLLs included in the rustc package
        // above) and the second argument is where to place all the MinGW components
        // (which is what we want).
        make_win_dist(&tmpdir(builder), &image, host, &builder);

        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust-MinGW")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=Rust-MinGW-is-installed.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg(format!("--package-name={}-{}", name, host.triple))
            .arg("--component-name=rust-mingw")
            .arg("--legacy-manifest-dirs=rustlib,cargo");
        builder.run(&mut cmd);
        t!(fs::remove_dir_all(&image));
        Some(distdir(builder).join(format!("{}-{}.tar.gz", name, host.triple)))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Rustc {
    pub compiler: Compiler,
}

impl Step for Rustc {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/librustc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder
            .ensure(Rustc { compiler: run.builder.compiler(run.builder.top_stage, run.target) });
    }

    /// Creates the `rustc` installer component.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let host = self.compiler.host;

        let name = pkgname(builder, "rustc");
        let image = tmpdir(builder).join(format!("{}-{}-image", name, host.triple));
        let _ = fs::remove_dir_all(&image);
        let overlay = tmpdir(builder).join(format!("{}-{}-overlay", name, host.triple));
        let _ = fs::remove_dir_all(&overlay);

        // Prepare the rustc "image", what will actually end up getting installed
        prepare_image(builder, compiler, &image);

        // Prepare the overlay which is part of the tarball but won't actually be
        // installed
        let cp = |file: &str| {
            builder.install(&builder.src.join(file), &overlay, 0o644);
        };
        cp("COPYRIGHT");
        cp("LICENSE-APACHE");
        cp("LICENSE-MIT");
        cp("README.md");
        // tiny morsel of metadata is used by rust-packaging
        let version = builder.rust_version();
        builder.create(&overlay.join("version"), &version);
        if let Some(sha) = builder.rust_sha() {
            builder.create(&overlay.join("git-commit-hash"), &sha);
        }

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
        if host.contains("pc-windows-gnu") {
            make_win_dist(&image, &tmpdir(builder), host, builder);

            let dst = image.join("share/doc");
            t!(fs::create_dir_all(&dst));
            builder.cp_r(&builder.src.join("src/etc/third-party"), &dst);
        }

        // Finally, wrap everything up in a nice tarball!
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=Rust-is-ready-to-roll.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, host.triple))
            .arg("--component-name=rustc")
            .arg("--legacy-manifest-dirs=rustlib,cargo");

        builder.info(&format!("Dist rustc stage{} ({})", compiler.stage, host.triple));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        builder.remove_dir(&image);
        builder.remove_dir(&overlay);

        return distdir(builder).join(format!("{}-{}.tar.gz", name, host.triple));

        fn prepare_image(builder: &Builder<'_>, compiler: Compiler, image: &Path) {
            let host = compiler.host;
            let src = builder.sysroot(compiler);

            // Copy rustc/rustdoc binaries
            t!(fs::create_dir_all(image.join("bin")));
            builder.cp_r(&src.join("bin"), &image.join("bin"));

            builder.install(&builder.rustdoc(compiler), &image.join("bin"), 0o755);

            let libdir_relative = builder.libdir_relative(compiler);

            // Copy runtime DLLs needed by the compiler
            if libdir_relative.to_str() != Some("bin") {
                let libdir = builder.rustc_libdir(compiler);
                for entry in builder.read_dir(&libdir) {
                    let name = entry.file_name();
                    if let Some(s) = name.to_str() {
                        if is_dylib(s) {
                            // Don't use custom libdir here because ^lib/ will be resolved again
                            // with installer
                            builder.install(&entry.path(), &image.join("lib"), 0o644);
                        }
                    }
                }
            }

            // Copy over the codegen backends
            let backends_src = builder.sysroot_codegen_backends(compiler);
            let backends_rel = backends_src
                .strip_prefix(&src)
                .unwrap()
                .strip_prefix(builder.sysroot_libdir_relative(compiler))
                .unwrap();
            // Don't use custom libdir here because ^lib/ will be resolved again with installer
            let backends_dst = image.join("lib").join(&backends_rel);

            t!(fs::create_dir_all(&backends_dst));
            builder.cp_r(&backends_src, &backends_dst);

            // Copy libLLVM.so to the lib dir as well, if needed. While not
            // technically needed by rustc itself it's needed by lots of other
            // components like the llvm tools and LLD. LLD is included below and
            // tools/LLDB come later, so let's just throw it in the rustc
            // component for now.
            maybe_install_llvm_runtime(builder, host, image);

            // Copy over lld if it's there
            if builder.config.lld_enabled {
                let exe = exe("rust-lld", compiler.host);
                let src =
                    builder.sysroot_libdir(compiler, host).parent().unwrap().join("bin").join(&exe);
                // for the rationale about this rename check `compile::copy_lld_to_sysroot`
                let dst = image.join("lib/rustlib").join(&*host.triple).join("bin").join(&exe);
                t!(fs::create_dir_all(&dst.parent().unwrap()));
                builder.copy(&src, &dst);
            }

            // Man pages
            t!(fs::create_dir_all(image.join("share/man/man1")));
            let man_src = builder.src.join("src/doc/man");
            let man_dst = image.join("share/man/man1");

            // Reproducible builds: If SOURCE_DATE_EPOCH is set, use that as the time.
            let time = env::var("SOURCE_DATE_EPOCH")
                .map(|timestamp| {
                    let epoch = timestamp
                        .parse()
                        .map_err(|err| format!("could not parse SOURCE_DATE_EPOCH: {}", err))
                        .unwrap();

                    time::at(Timespec::new(epoch, 0))
                })
                .unwrap_or_else(|_| time::now());

            let month_year = t!(time::strftime("%B %Y", &time));
            // don't use our `bootstrap::util::{copy, cp_r}`, because those try
            // to hardlink, and we don't want to edit the source templates
            for file_entry in builder.read_dir(&man_src) {
                let page_src = file_entry.path();
                let page_dst = man_dst.join(file_entry.file_name());
                t!(fs::copy(&page_src, &page_dst));
                // template in month/year and version number
                builder.replace_in_file(
                    &page_dst,
                    &[
                        ("<INSERT DATE HERE>", &month_year),
                        ("<INSERT VERSION HERE>", &builder.version),
                    ],
                );
            }

            // Debugger scripts
            builder
                .ensure(DebuggerScripts { sysroot: INTERNER.intern_path(image.to_owned()), host });

            // Misc license info
            let cp = |file: &str| {
                builder.install(&builder.src.join(file), &image.join("share/doc/rust"), 0o644);
            };
            cp("COPYRIGHT");
            cp("LICENSE-APACHE");
            cp("LICENSE-MIT");
            cp("README.md");
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct DebuggerScripts {
    pub sysroot: Interned<PathBuf>,
    pub host: TargetSelection,
}

impl Step for DebuggerScripts {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/lldb_batchmode.py")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(DebuggerScripts {
            sysroot: run
                .builder
                .sysroot(run.builder.compiler(run.builder.top_stage, run.build_triple())),
            host: run.target,
        });
    }

    /// Copies debugger scripts for `target` into the `sysroot` specified.
    fn run(self, builder: &Builder<'_>) {
        let host = self.host;
        let sysroot = self.sysroot;
        let dst = sysroot.join("lib/rustlib/etc");
        t!(fs::create_dir_all(&dst));
        let cp_debugger_script = |file: &str| {
            builder.install(&builder.src.join("src/etc/").join(file), &dst, 0o644);
        };
        if host.contains("windows-msvc") {
            // windbg debugger scripts
            builder.install(
                &builder.src.join("src/etc/rust-windbg.cmd"),
                &sysroot.join("bin"),
                0o755,
            );

            cp_debugger_script("natvis/intrinsic.natvis");
            cp_debugger_script("natvis/liballoc.natvis");
            cp_debugger_script("natvis/libcore.natvis");
            cp_debugger_script("natvis/libstd.natvis");
        } else {
            cp_debugger_script("rust_types.py");

            // gdb debugger scripts
            builder.install(&builder.src.join("src/etc/rust-gdb"), &sysroot.join("bin"), 0o755);
            builder.install(&builder.src.join("src/etc/rust-gdbgui"), &sysroot.join("bin"), 0o755);

            cp_debugger_script("gdb_load_rust_pretty_printers.py");
            cp_debugger_script("gdb_lookup.py");
            cp_debugger_script("gdb_providers.py");

            // lldb debugger scripts
            builder.install(&builder.src.join("src/etc/rust-lldb"), &sysroot.join("bin"), 0o755);

            cp_debugger_script("lldb_lookup.py");
            cp_debugger_script("lldb_providers.py");
            cp_debugger_script("lldb_commands")
        }
    }
}

fn skip_host_target_lib(builder: &Builder<'_>, compiler: Compiler) -> bool {
    // The only true set of target libraries came from the build triple, so
    // let's reduce redundant work by only producing archives from that host.
    if compiler.host != builder.config.build {
        builder.info("\tskipping, not a build host");
        true
    } else {
        false
    }
}

/// Copy stamped files into an image's `target/lib` directory.
fn copy_target_libs(builder: &Builder<'_>, target: TargetSelection, image: &Path, stamp: &Path) {
    let dst = image.join("lib/rustlib").join(target.triple).join("lib");
    let self_contained_dst = dst.join("self-contained");
    t!(fs::create_dir_all(&dst));
    t!(fs::create_dir_all(&self_contained_dst));
    for (path, dependency_type) in builder.read_stamp_file(stamp) {
        if dependency_type == DependencyType::TargetSelfContained {
            builder.copy(&path, &self_contained_dst.join(path.file_name().unwrap()));
        } else if dependency_type == DependencyType::Target || builder.config.build == target {
            builder.copy(&path, &dst.join(path.file_name().unwrap()));
        }
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Std {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Std {
    type Output = PathBuf;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("library/std")
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

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let target = self.target;

        let name = pkgname(builder, "rust-std");
        let archive = distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple));
        if skip_host_target_lib(builder, compiler) {
            return archive;
        }

        builder.ensure(compile::Std { compiler, target });

        let image = tmpdir(builder).join(format!("{}-{}-image", name, target.triple));
        let _ = fs::remove_dir_all(&image);

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        let stamp = compile::libstd_stamp(builder, compiler_to_use, target);
        copy_target_libs(builder, target, &image, &stamp);

        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=std-is-standing-at-the-ready.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg(format!("--component-name=rust-std-{}", target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo");

        builder
            .info(&format!("Dist std stage{} ({} -> {})", compiler.stage, &compiler.host, target));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        builder.remove_dir(&image);
        archive
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustcDev {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustcDev {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("rustc-dev")
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

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let target = self.target;

        let name = pkgname(builder, "rustc-dev");
        let archive = distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple));
        if skip_host_target_lib(builder, compiler) {
            return archive;
        }

        builder.ensure(compile::Rustc { compiler, target });

        let image = tmpdir(builder).join(format!("{}-{}-image", name, target.triple));
        let _ = fs::remove_dir_all(&image);

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        let stamp = compile::librustc_stamp(builder, compiler_to_use, target);
        copy_target_libs(builder, target, &image, &stamp);

        // Copy compiler sources.
        let dst_src = image.join("lib/rustlib/rustc-src/rust");
        t!(fs::create_dir_all(&dst_src));

        let src_files = ["Cargo.lock"];
        // This is the reduced set of paths which will become the rustc-dev component
        // (essentially the compiler crates and all of their path dependencies).
        copy_src_dirs(builder, &builder.src, &["compiler"], &[], &dst_src);
        for file in src_files.iter() {
            builder.copy(&builder.src.join(file), &dst_src.join(file));
        }

        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=Rust-is-ready-to-develop.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg(format!("--component-name=rustc-dev-{}", target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo");

        builder.info(&format!(
            "Dist rustc-dev stage{} ({} -> {})",
            compiler.stage, &compiler.host, target
        ));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        builder.remove_dir(&image);
        archive
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Analysis {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Analysis {
    type Output = PathBuf;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("analysis").default_condition(builder.config.extended)
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

    /// Creates a tarball of save-analysis metadata, if available.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let target = self.target;
        assert!(builder.config.extended);
        let name = pkgname(builder, "rust-analysis");

        if compiler.host != builder.config.build {
            return distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple));
        }

        builder.ensure(compile::Std { compiler, target });

        let image = tmpdir(builder).join(format!("{}-{}-image", name, target.triple));

        let src = builder
            .stage_out(compiler, Mode::Std)
            .join(target.triple)
            .join(builder.cargo_dir())
            .join("deps");

        let image_src = src.join("save-analysis");
        let dst = image.join("lib/rustlib").join(target.triple).join("analysis");
        t!(fs::create_dir_all(&dst));
        builder.info(&format!("image_src: {:?}, dst: {:?}", image_src, dst));
        builder.cp_r(&image_src, &dst);

        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=save-analysis-saved.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg(format!("--component-name=rust-analysis-{}", target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo");

        builder.info("Dist analysis");
        let _time = timeit(builder);
        builder.run(&mut cmd);
        builder.remove_dir(&image);
        distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple))
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
        builder.cp_filtered(&base.join(item), dst, &|path| filter_fn(exclude_dirs, item, path));
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Src;

impl Step for Src {
    /// The output path of the src installer tarball
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Src);
    }

    /// Creates the `rust-src` installer component
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let name = pkgname(builder, "rust-src");
        let image = tmpdir(builder).join(format!("{}-image", name));
        let _ = fs::remove_dir_all(&image);

        // A lot of tools expect the rust-src component to be entirely in this directory, so if you
        // change that (e.g. by adding another directory `lib/rustlib/src/foo` or
        // `lib/rustlib/src/rust/foo`), you will need to go around hunting for implicit assumptions
        // and fix them...
        //
        // NOTE: if you update the paths here, you also should update the "virtual" path
        // translation code in `imported_source_files` in `src/librustc_metadata/rmeta/decoder.rs`
        let dst_src = image.join("lib/rustlib/src/rust");
        t!(fs::create_dir_all(&dst_src));

        let src_files = ["Cargo.lock"];
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
            ],
            &dst_src,
        );
        for file in src_files.iter() {
            builder.copy(&builder.src.join(file), &dst_src.join(file));
        }

        // Create source tarball in rust-installer format
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=Awesome-Source.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg(format!("--package-name={}", name))
            .arg("--component-name=rust-src")
            .arg("--legacy-manifest-dirs=rustlib,cargo");

        builder.info("Dist src");
        let _time = timeit(builder);
        builder.run(&mut cmd);

        builder.remove_dir(&image);
        distdir(builder).join(&format!("{}.tar.gz", name))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct PlainSourceTarball;

impl Step for PlainSourceTarball {
    /// Produces the location of the tarball generated
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src").default_condition(builder.config.rust_dist_src)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(PlainSourceTarball);
    }

    /// Creates the plain source tarball
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        // Make sure that the root folder of tarball has the correct name
        let plain_name = format!("{}-src", pkgname(builder, "rustc"));
        let plain_dst_src = tmpdir(builder).join(&plain_name);
        let _ = fs::remove_dir_all(&plain_dst_src);
        t!(fs::create_dir_all(&plain_dst_src));

        // This is the set of root paths which will become part of the source package
        let src_files = [
            "COPYRIGHT",
            "LICENSE-APACHE",
            "LICENSE-MIT",
            "CONTRIBUTING.md",
            "README.md",
            "RELEASES.md",
            "configure",
            "x.py",
            "config.toml.example",
            "Cargo.toml",
            "Cargo.lock",
        ];
        let src_dirs = ["src", "compiler", "library"];

        copy_src_dirs(builder, &builder.src, &src_dirs, &[], &plain_dst_src);

        // Copy the files normally
        for item in &src_files {
            builder.copy(&builder.src.join(item), &plain_dst_src.join(item));
        }

        // Create the version file
        builder.create(&plain_dst_src.join("version"), &builder.rust_version());
        if let Some(sha) = builder.rust_sha() {
            builder.create(&plain_dst_src.join("git-commit-hash"), &sha);
        }

        // If we're building from git sources, we need to vendor a complete distribution.
        if builder.rust_info.is_git() {
            // Vendor all Cargo dependencies
            let mut cmd = Command::new(&builder.initial_cargo);
            cmd.arg("vendor")
                .arg("--sync")
                .arg(builder.src.join("./src/tools/rust-analyzer/Cargo.toml"))
                .arg(builder.src.join("./compiler/rustc_codegen_cranelift/Cargo.toml"))
                .current_dir(&plain_dst_src);
            builder.run(&mut cmd);
        }

        // Create plain source tarball
        let plain_name = format!("rustc-{}-src", builder.rust_package_vers());
        let mut tarball = distdir(builder).join(&format!("{}.tar.gz", plain_name));
        tarball.set_extension(""); // strip .gz
        tarball.set_extension(""); // strip .tar
        if let Some(dir) = tarball.parent() {
            builder.create_dir(&dir);
        }
        builder.info("running installer");
        let mut cmd = rust_installer(builder);
        cmd.arg("tarball")
            .arg("--input")
            .arg(&plain_name)
            .arg("--output")
            .arg(&tarball)
            .arg("--work-dir=.")
            .current_dir(tmpdir(builder));

        builder.info("Create plain source tarball");
        let _time = timeit(builder);
        builder.run(&mut cmd);
        distdir(builder).join(&format!("{}.tar.gz", plain_name))
    }
}

// We have to run a few shell scripts, which choke quite a bit on both `\`
// characters and on `C:\` paths, so normalize both of them away.
pub fn sanitize_sh(path: &Path) -> String {
    let path = path.to_str().unwrap().replace("\\", "/");
    return change_drive(&path).unwrap_or(path);

    fn change_drive(s: &str) -> Option<String> {
        let mut ch = s.chars();
        let drive = ch.next().unwrap_or('C');
        if ch.next() != Some(':') {
            return None;
        }
        if ch.next() != Some('/') {
            return None;
        }
        Some(format!("/{}/{}", drive, &s[drive.len_utf8() + 2..]))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Cargo {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Cargo {
    type Output = PathBuf;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("cargo")
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

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let target = self.target;

        let src = builder.src.join("src/tools/cargo");
        let etc = src.join("src/etc");
        let release_num = builder.release_num("cargo");
        let name = pkgname(builder, "cargo");
        let version = builder.cargo_info.version(builder, &release_num);

        let tmp = tmpdir(builder);
        let image = tmp.join("cargo-image");
        drop(fs::remove_dir_all(&image));
        builder.create_dir(&image);

        // Prepare the image directory
        builder.create_dir(&image.join("share/zsh/site-functions"));
        builder.create_dir(&image.join("etc/bash_completion.d"));
        let cargo = builder.ensure(tool::Cargo { compiler, target });
        builder.install(&cargo, &image.join("bin"), 0o755);
        for man in t!(etc.join("man").read_dir()) {
            let man = t!(man);
            builder.install(&man.path(), &image.join("share/man/man1"), 0o644);
        }
        builder.install(&etc.join("_cargo"), &image.join("share/zsh/site-functions"), 0o644);
        builder.copy(&etc.join("cargo.bashcomp.sh"), &image.join("etc/bash_completion.d/cargo"));
        let doc = image.join("share/doc/cargo");
        builder.install(&src.join("README.md"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);
        builder.install(&src.join("LICENSE-THIRD-PARTY"), &doc, 0o644);

        // Prepare the overlay
        let overlay = tmp.join("cargo-overlay");
        drop(fs::remove_dir_all(&overlay));
        builder.create_dir(&overlay);
        builder.install(&src.join("README.md"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-THIRD-PARTY"), &overlay, 0o644);
        builder.create(&overlay.join("version"), &version);

        // Generate the installer tarball
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=Rust-is-ready-to-roll.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg("--component-name=cargo")
            .arg("--legacy-manifest-dirs=rustlib,cargo");

        builder.info(&format!("Dist cargo stage{} ({})", compiler.stage, target));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Rls {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Rls {
    type Output = Option<PathBuf>;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("rls")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rls {
            compiler: run.builder.compiler_for(
                run.builder.top_stage,
                run.builder.config.build,
                run.target,
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<PathBuf> {
        let compiler = self.compiler;
        let target = self.target;
        assert!(builder.config.extended);

        let src = builder.src.join("src/tools/rls");
        let release_num = builder.release_num("rls");
        let name = pkgname(builder, "rls");
        let version = builder.rls_info.version(builder, &release_num);

        let tmp = tmpdir(builder);
        let image = tmp.join("rls-image");
        drop(fs::remove_dir_all(&image));
        t!(fs::create_dir_all(&image));

        // Prepare the image directory
        // We expect RLS to build, because we've exited this step above if tool
        // state for RLS isn't testing.
        let rls = builder
            .ensure(tool::Rls { compiler, target, extra_features: Vec::new() })
            .or_else(|| {
                missing_tool("RLS", builder.build.config.missing_tools);
                None
            })?;

        builder.install(&rls, &image.join("bin"), 0o755);
        let doc = image.join("share/doc/rls");
        builder.install(&src.join("README.md"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);

        // Prepare the overlay
        let overlay = tmp.join("rls-overlay");
        drop(fs::remove_dir_all(&overlay));
        t!(fs::create_dir_all(&overlay));
        builder.install(&src.join("README.md"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &overlay, 0o644);
        builder.create(&overlay.join("version"), &version);

        // Generate the installer tarball
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=RLS-ready-to-serve.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--component-name=rls-preview");

        builder.info(&format!("Dist RLS stage{} ({})", compiler.stage, target.triple));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        Some(distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple)))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RustAnalyzer {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustAnalyzer {
    type Output = Option<PathBuf>;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("rust-analyzer")
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

    fn run(self, builder: &Builder<'_>) -> Option<PathBuf> {
        let compiler = self.compiler;
        let target = self.target;
        assert!(builder.config.extended);

        if target.contains("riscv64") {
            // riscv64 currently has an LLVM bug that makes rust-analyzer unable
            // to build. See #74813 for details.
            return None;
        }

        let src = builder.src.join("src/tools/rust-analyzer");
        let release_num = builder.release_num("rust-analyzer/crates/rust-analyzer");
        let name = pkgname(builder, "rust-analyzer");
        let version = builder.rust_analyzer_info.version(builder, &release_num);

        let tmp = tmpdir(builder);
        let image = tmp.join("rust-analyzer-image");
        drop(fs::remove_dir_all(&image));
        builder.create_dir(&image);

        // Prepare the image directory
        // We expect rust-analyer to always build, as it doesn't depend on rustc internals
        // and doesn't have associated toolstate.
        let rust_analyzer = builder
            .ensure(tool::RustAnalyzer { compiler, target, extra_features: Vec::new() })
            .expect("rust-analyzer always builds");

        builder.install(&rust_analyzer, &image.join("bin"), 0o755);
        let doc = image.join("share/doc/rust-analyzer");
        builder.install(&src.join("README.md"), &doc, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);

        // Prepare the overlay
        let overlay = tmp.join("rust-analyzer-overlay");
        drop(fs::remove_dir_all(&overlay));
        t!(fs::create_dir_all(&overlay));
        builder.install(&src.join("README.md"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);
        builder.create(&overlay.join("version"), &version);

        // Generate the installer tarball
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=rust-analyzer-ready-to-serve.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--component-name=rust-analyzer-preview");

        builder.info(&format!("Dist rust-analyzer stage{} ({})", compiler.stage, target));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        Some(distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple)))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Clippy {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Clippy {
    type Output = PathBuf;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("clippy")
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

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let target = self.target;
        assert!(builder.config.extended);

        let src = builder.src.join("src/tools/clippy");
        let release_num = builder.release_num("clippy");
        let name = pkgname(builder, "clippy");
        let version = builder.clippy_info.version(builder, &release_num);

        let tmp = tmpdir(builder);
        let image = tmp.join("clippy-image");
        drop(fs::remove_dir_all(&image));
        builder.create_dir(&image);

        // Prepare the image directory
        // We expect clippy to build, because we've exited this step above if tool
        // state for clippy isn't testing.
        let clippy = builder
            .ensure(tool::Clippy { compiler, target, extra_features: Vec::new() })
            .expect("clippy expected to build - essential tool");
        let cargoclippy = builder
            .ensure(tool::CargoClippy { compiler, target, extra_features: Vec::new() })
            .expect("clippy expected to build - essential tool");

        builder.install(&clippy, &image.join("bin"), 0o755);
        builder.install(&cargoclippy, &image.join("bin"), 0o755);
        let doc = image.join("share/doc/clippy");
        builder.install(&src.join("README.md"), &doc, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);

        // Prepare the overlay
        let overlay = tmp.join("clippy-overlay");
        drop(fs::remove_dir_all(&overlay));
        t!(fs::create_dir_all(&overlay));
        builder.install(&src.join("README.md"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);
        builder.create(&overlay.join("version"), &version);

        // Generate the installer tarball
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=clippy-ready-to-serve.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--component-name=clippy-preview");

        builder.info(&format!("Dist clippy stage{} ({})", compiler.stage, target));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Miri {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Miri {
    type Output = Option<PathBuf>;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("miri")
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

    fn run(self, builder: &Builder<'_>) -> Option<PathBuf> {
        let compiler = self.compiler;
        let target = self.target;
        assert!(builder.config.extended);

        let src = builder.src.join("src/tools/miri");
        let release_num = builder.release_num("miri");
        let name = pkgname(builder, "miri");
        let version = builder.miri_info.version(builder, &release_num);

        let tmp = tmpdir(builder);
        let image = tmp.join("miri-image");
        drop(fs::remove_dir_all(&image));
        builder.create_dir(&image);

        // Prepare the image directory
        // We expect miri to build, because we've exited this step above if tool
        // state for miri isn't testing.
        let miri = builder
            .ensure(tool::Miri { compiler, target, extra_features: Vec::new() })
            .or_else(|| {
                missing_tool("miri", builder.build.config.missing_tools);
                None
            })?;
        let cargomiri = builder
            .ensure(tool::CargoMiri { compiler, target, extra_features: Vec::new() })
            .or_else(|| {
                missing_tool("cargo miri", builder.build.config.missing_tools);
                None
            })?;

        builder.install(&miri, &image.join("bin"), 0o755);
        builder.install(&cargomiri, &image.join("bin"), 0o755);
        let doc = image.join("share/doc/miri");
        builder.install(&src.join("README.md"), &doc, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);

        // Prepare the overlay
        let overlay = tmp.join("miri-overlay");
        drop(fs::remove_dir_all(&overlay));
        t!(fs::create_dir_all(&overlay));
        builder.install(&src.join("README.md"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);
        builder.create(&overlay.join("version"), &version);

        // Generate the installer tarball
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=miri-ready-to-serve.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--component-name=miri-preview");

        builder.info(&format!("Dist miri stage{} ({})", compiler.stage, target));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        Some(distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple)))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Rustfmt {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Rustfmt {
    type Output = Option<PathBuf>;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("rustfmt")
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

    fn run(self, builder: &Builder<'_>) -> Option<PathBuf> {
        let compiler = self.compiler;
        let target = self.target;

        let src = builder.src.join("src/tools/rustfmt");
        let release_num = builder.release_num("rustfmt");
        let name = pkgname(builder, "rustfmt");
        let version = builder.rustfmt_info.version(builder, &release_num);

        let tmp = tmpdir(builder);
        let image = tmp.join("rustfmt-image");
        drop(fs::remove_dir_all(&image));
        builder.create_dir(&image);

        // Prepare the image directory
        let rustfmt = builder
            .ensure(tool::Rustfmt { compiler, target, extra_features: Vec::new() })
            .or_else(|| {
                missing_tool("Rustfmt", builder.build.config.missing_tools);
                None
            })?;
        let cargofmt = builder
            .ensure(tool::Cargofmt { compiler, target, extra_features: Vec::new() })
            .or_else(|| {
                missing_tool("Cargofmt", builder.build.config.missing_tools);
                None
            })?;

        builder.install(&rustfmt, &image.join("bin"), 0o755);
        builder.install(&cargofmt, &image.join("bin"), 0o755);
        let doc = image.join("share/doc/rustfmt");
        builder.install(&src.join("README.md"), &doc, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &doc, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &doc, 0o644);

        // Prepare the overlay
        let overlay = tmp.join("rustfmt-overlay");
        drop(fs::remove_dir_all(&overlay));
        builder.create_dir(&overlay);
        builder.install(&src.join("README.md"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-MIT"), &overlay, 0o644);
        builder.install(&src.join("LICENSE-APACHE"), &overlay, 0o644);
        builder.create(&overlay.join("version"), &version);

        // Generate the installer tarball
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=rustfmt-ready-to-fmt.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--component-name=rustfmt-preview");

        builder.info(&format!("Dist Rustfmt stage{} ({})", compiler.stage, target));
        let _time = timeit(builder);
        builder.run(&mut cmd);
        Some(distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple)))
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
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
        run.path("extended").default_condition(builder.config.extended)
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

        let rustc_installer = builder.ensure(Rustc { compiler: builder.compiler(stage, target) });
        let cargo_installer = builder.ensure(Cargo { compiler, target });
        let rustfmt_installer = builder.ensure(Rustfmt { compiler, target });
        let rls_installer = builder.ensure(Rls { compiler, target });
        let rust_analyzer_installer = builder.ensure(RustAnalyzer { compiler, target });
        let llvm_tools_installer = builder.ensure(LlvmTools { target });
        let clippy_installer = builder.ensure(Clippy { compiler, target });
        let miri_installer = builder.ensure(Miri { compiler, target });
        let mingw_installer = builder.ensure(Mingw { host: target });
        let analysis_installer = builder.ensure(Analysis { compiler, target });

        let docs_installer = builder.ensure(Docs { host: target });
        let std_installer =
            builder.ensure(Std { compiler: builder.compiler(stage, target), target });

        let tmp = tmpdir(builder);
        let overlay = tmp.join("extended-overlay");
        let etc = builder.src.join("src/etc/installer");
        let work = tmp.join("work");

        let _ = fs::remove_dir_all(&overlay);
        builder.install(&builder.src.join("COPYRIGHT"), &overlay, 0o644);
        builder.install(&builder.src.join("LICENSE-APACHE"), &overlay, 0o644);
        builder.install(&builder.src.join("LICENSE-MIT"), &overlay, 0o644);
        let version = builder.rust_version();
        builder.create(&overlay.join("version"), &version);
        if let Some(sha) = builder.rust_sha() {
            builder.create(&overlay.join("git-commit-hash"), &sha);
        }
        builder.install(&etc.join("README.md"), &overlay, 0o644);

        // When rust-std package split from rustc, we needed to ensure that during
        // upgrades rustc was upgraded before rust-std. To avoid rustc clobbering
        // the std files during uninstall. To do this ensure that rustc comes
        // before rust-std in the list below.
        let mut tarballs = Vec::new();
        tarballs.push(rustc_installer);
        tarballs.push(cargo_installer);
        tarballs.extend(rls_installer.clone());
        tarballs.extend(rust_analyzer_installer.clone());
        tarballs.push(clippy_installer);
        tarballs.extend(miri_installer.clone());
        tarballs.extend(rustfmt_installer.clone());
        tarballs.extend(llvm_tools_installer);
        tarballs.push(analysis_installer);
        tarballs.push(std_installer);
        if builder.config.docs {
            tarballs.push(docs_installer);
        }
        if target.contains("pc-windows-gnu") {
            tarballs.push(mingw_installer.unwrap());
        }
        let mut input_tarballs = tarballs[0].as_os_str().to_owned();
        for tarball in &tarballs[1..] {
            input_tarballs.push(",");
            input_tarballs.push(tarball);
        }

        builder.info("building combined installer");
        let mut cmd = rust_installer(builder);
        cmd.arg("combine")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=Rust-is-ready-to-roll.")
            .arg("--work-dir")
            .arg(&work)
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg(format!("--package-name={}-{}", pkgname(builder, "rust"), target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--input-tarballs")
            .arg(input_tarballs)
            .arg("--non-installed-overlay")
            .arg(&overlay);
        let time = timeit(&builder);
        builder.run(&mut cmd);
        drop(time);

        let mut license = String::new();
        license += &builder.read(&builder.src.join("COPYRIGHT"));
        license += &builder.read(&builder.src.join("LICENSE-APACHE"));
        license += &builder.read(&builder.src.join("LICENSE-MIT"));
        license.push_str("\n");
        license.push_str("\n");

        let rtf = r"{\rtf1\ansi\deff0{\fonttbl{\f0\fnil\fcharset0 Arial;}}\nowwrap\fs18";
        let mut rtf = rtf.to_string();
        rtf.push_str("\n");
        for line in license.lines() {
            rtf.push_str(line);
            rtf.push_str("\\line ");
        }
        rtf.push_str("}");

        fn filter(contents: &str, marker: &str) -> String {
            let start = format!("tool-{}-start", marker);
            let end = format!("tool-{}-end", marker);
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
            if rls_installer.is_none() {
                contents = filter(&contents, "rls");
            }
            if rust_analyzer_installer.is_none() {
                contents = filter(&contents, "rust-analyzer");
            }
            if miri_installer.is_none() {
                contents = filter(&contents, "miri");
            }
            if rustfmt_installer.is_none() {
                contents = filter(&contents, "rustfmt");
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
                let mut cmd = Command::new("pkgbuild");
                cmd.arg("--identifier")
                    .arg(format!("org.rust-lang.{}", component))
                    .arg("--scripts")
                    .arg(pkg.join(component))
                    .arg("--nopayload")
                    .arg(pkg.join(component).with_extension("pkg"));
                builder.run(&mut cmd);
            };

            let prepare = |name: &str| {
                builder.create_dir(&pkg.join(name));
                builder.cp_r(
                    &work.join(&format!("{}-{}", pkgname(builder, name), target.triple)),
                    &pkg.join(name),
                );
                builder.install(&etc.join("pkg/postinstall"), &pkg.join(name), 0o755);
                pkgbuild(name);
            };
            prepare("rustc");
            prepare("cargo");
            prepare("rust-docs");
            prepare("rust-std");
            prepare("rust-analysis");
            prepare("clippy");

            if rls_installer.is_some() {
                prepare("rls");
            }
            if rust_analyzer_installer.is_some() {
                prepare("rust-analyzer");
            }
            if miri_installer.is_some() {
                prepare("miri");
            }

            // create an 'uninstall' package
            builder.install(&etc.join("pkg/postinstall"), &pkg.join("uninstall"), 0o755);
            pkgbuild("uninstall");

            builder.create_dir(&pkg.join("res"));
            builder.create(&pkg.join("res/LICENSE.txt"), &license);
            builder.install(&etc.join("gfx/rust-logo.png"), &pkg.join("res"), 0o644);
            let mut cmd = Command::new("productbuild");
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
            builder.run(&mut cmd);
        }

        if target.contains("windows") {
            let exe = tmp.join("exe");
            let _ = fs::remove_dir_all(&exe);

            let prepare = |name: &str| {
                builder.create_dir(&exe.join(name));
                let dir = if name == "rust-std" || name == "rust-analysis" {
                    format!("{}-{}", name, target.triple)
                } else if name == "rls" {
                    "rls-preview".to_string()
                } else if name == "rust-analyzer" {
                    "rust-analyzer-preview".to_string()
                } else if name == "clippy" {
                    "clippy-preview".to_string()
                } else if name == "miri" {
                    "miri-preview".to_string()
                } else {
                    name.to_string()
                };
                builder.cp_r(
                    &work.join(&format!("{}-{}", pkgname(builder, name), target.triple)).join(dir),
                    &exe.join(name),
                );
                builder.remove(&exe.join(name).join("manifest.in"));
            };
            prepare("rustc");
            prepare("cargo");
            prepare("rust-analysis");
            prepare("rust-docs");
            prepare("rust-std");
            prepare("clippy");
            if rls_installer.is_some() {
                prepare("rls");
            }
            if rust_analyzer_installer.is_some() {
                prepare("rust-analyzer");
            }
            if miri_installer.is_some() {
                prepare("miri");
            }
            if target.contains("windows-gnu") {
                prepare("rust-mingw");
            }

            builder.install(&etc.join("gfx/rust-logo.ico"), &exe, 0o644);

            // Generate msi installer
            let wix = PathBuf::from(env::var_os("WIX").unwrap());
            let heat = wix.join("bin/heat.exe");
            let candle = wix.join("bin/candle.exe");
            let light = wix.join("bin/light.exe");

            let heat_flags = ["-nologo", "-gg", "-sfrag", "-srd", "-sreg"];
            builder.run(
                Command::new(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("rustc")
                    .args(&heat_flags)
                    .arg("-cg")
                    .arg("RustcGroup")
                    .arg("-dr")
                    .arg("Rustc")
                    .arg("-var")
                    .arg("var.RustcDir")
                    .arg("-out")
                    .arg(exe.join("RustcGroup.wxs")),
            );
            builder.run(
                Command::new(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("rust-docs")
                    .args(&heat_flags)
                    .arg("-cg")
                    .arg("DocsGroup")
                    .arg("-dr")
                    .arg("Docs")
                    .arg("-var")
                    .arg("var.DocsDir")
                    .arg("-out")
                    .arg(exe.join("DocsGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/squash-components.xsl")),
            );
            builder.run(
                Command::new(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("cargo")
                    .args(&heat_flags)
                    .arg("-cg")
                    .arg("CargoGroup")
                    .arg("-dr")
                    .arg("Cargo")
                    .arg("-var")
                    .arg("var.CargoDir")
                    .arg("-out")
                    .arg(exe.join("CargoGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/remove-duplicates.xsl")),
            );
            builder.run(
                Command::new(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("rust-std")
                    .args(&heat_flags)
                    .arg("-cg")
                    .arg("StdGroup")
                    .arg("-dr")
                    .arg("Std")
                    .arg("-var")
                    .arg("var.StdDir")
                    .arg("-out")
                    .arg(exe.join("StdGroup.wxs")),
            );
            if rls_installer.is_some() {
                builder.run(
                    Command::new(&heat)
                        .current_dir(&exe)
                        .arg("dir")
                        .arg("rls")
                        .args(&heat_flags)
                        .arg("-cg")
                        .arg("RlsGroup")
                        .arg("-dr")
                        .arg("Rls")
                        .arg("-var")
                        .arg("var.RlsDir")
                        .arg("-out")
                        .arg(exe.join("RlsGroup.wxs"))
                        .arg("-t")
                        .arg(etc.join("msi/remove-duplicates.xsl")),
                );
            }
            if rust_analyzer_installer.is_some() {
                builder.run(
                    Command::new(&heat)
                        .current_dir(&exe)
                        .arg("dir")
                        .arg("rust-analyzer")
                        .args(&heat_flags)
                        .arg("-cg")
                        .arg("RustAnalyzerGroup")
                        .arg("-dr")
                        .arg("RustAnalyzer")
                        .arg("-var")
                        .arg("var.RustAnalyzerDir")
                        .arg("-out")
                        .arg(exe.join("RustAnalyzerGroup.wxs"))
                        .arg("-t")
                        .arg(etc.join("msi/remove-duplicates.xsl")),
                );
            }
            builder.run(
                Command::new(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("clippy")
                    .args(&heat_flags)
                    .arg("-cg")
                    .arg("ClippyGroup")
                    .arg("-dr")
                    .arg("Clippy")
                    .arg("-var")
                    .arg("var.ClippyDir")
                    .arg("-out")
                    .arg(exe.join("ClippyGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/remove-duplicates.xsl")),
            );
            if miri_installer.is_some() {
                builder.run(
                    Command::new(&heat)
                        .current_dir(&exe)
                        .arg("dir")
                        .arg("miri")
                        .args(&heat_flags)
                        .arg("-cg")
                        .arg("MiriGroup")
                        .arg("-dr")
                        .arg("Miri")
                        .arg("-var")
                        .arg("var.MiriDir")
                        .arg("-out")
                        .arg(exe.join("MiriGroup.wxs"))
                        .arg("-t")
                        .arg(etc.join("msi/remove-duplicates.xsl")),
                );
            }
            builder.run(
                Command::new(&heat)
                    .current_dir(&exe)
                    .arg("dir")
                    .arg("rust-analysis")
                    .args(&heat_flags)
                    .arg("-cg")
                    .arg("AnalysisGroup")
                    .arg("-dr")
                    .arg("Analysis")
                    .arg("-var")
                    .arg("var.AnalysisDir")
                    .arg("-out")
                    .arg(exe.join("AnalysisGroup.wxs"))
                    .arg("-t")
                    .arg(etc.join("msi/remove-duplicates.xsl")),
            );
            if target.contains("windows-gnu") {
                builder.run(
                    Command::new(&heat)
                        .current_dir(&exe)
                        .arg("dir")
                        .arg("rust-mingw")
                        .args(&heat_flags)
                        .arg("-cg")
                        .arg("GccGroup")
                        .arg("-dr")
                        .arg("Gcc")
                        .arg("-var")
                        .arg("var.GccDir")
                        .arg("-out")
                        .arg(exe.join("GccGroup.wxs")),
                );
            }

            let candle = |input: &Path| {
                let output = exe.join(input.file_stem().unwrap()).with_extension("wixobj");
                let arch = if target.contains("x86_64") { "x64" } else { "x86" };
                let mut cmd = Command::new(&candle);
                cmd.current_dir(&exe)
                    .arg("-nologo")
                    .arg("-dRustcDir=rustc")
                    .arg("-dDocsDir=rust-docs")
                    .arg("-dCargoDir=cargo")
                    .arg("-dStdDir=rust-std")
                    .arg("-dAnalysisDir=rust-analysis")
                    .arg("-dClippyDir=clippy")
                    .arg("-arch")
                    .arg(&arch)
                    .arg("-out")
                    .arg(&output)
                    .arg(&input);
                add_env(builder, &mut cmd, target);

                if rls_installer.is_some() {
                    cmd.arg("-dRlsDir=rls");
                }
                if rust_analyzer_installer.is_some() {
                    cmd.arg("-dRustAnalyzerDir=rust-analyzer");
                }
                if miri_installer.is_some() {
                    cmd.arg("-dMiriDir=miri");
                }
                if target.contains("windows-gnu") {
                    cmd.arg("-dGccDir=rust-mingw");
                }
                builder.run(&mut cmd);
            };
            candle(&xform(&etc.join("msi/rust.wxs")));
            candle(&etc.join("msi/ui.wxs"));
            candle(&etc.join("msi/rustwelcomedlg.wxs"));
            candle("RustcGroup.wxs".as_ref());
            candle("DocsGroup.wxs".as_ref());
            candle("CargoGroup.wxs".as_ref());
            candle("StdGroup.wxs".as_ref());
            candle("ClippyGroup.wxs".as_ref());
            if rls_installer.is_some() {
                candle("RlsGroup.wxs".as_ref());
            }
            if rust_analyzer_installer.is_some() {
                candle("RustAnalyzerGroup.wxs".as_ref());
            }
            if miri_installer.is_some() {
                candle("MiriGroup.wxs".as_ref());
            }
            candle("AnalysisGroup.wxs".as_ref());

            if target.contains("windows-gnu") {
                candle("GccGroup.wxs".as_ref());
            }

            builder.create(&exe.join("LICENSE.rtf"), &rtf);
            builder.install(&etc.join("gfx/banner.bmp"), &exe, 0o644);
            builder.install(&etc.join("gfx/dialogbg.bmp"), &exe, 0o644);

            builder.info(&format!("building `msi` installer with {:?}", light));
            let filename = format!("{}-{}.msi", pkgname(builder, "rust"), target.triple);
            let mut cmd = Command::new(&light);
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
                .arg("DocsGroup.wixobj")
                .arg("CargoGroup.wixobj")
                .arg("StdGroup.wixobj")
                .arg("AnalysisGroup.wixobj")
                .arg("ClippyGroup.wixobj")
                .current_dir(&exe);

            if rls_installer.is_some() {
                cmd.arg("RlsGroup.wixobj");
            }
            if rust_analyzer_installer.is_some() {
                cmd.arg("RustAnalyzerGroup.wixobj");
            }
            if miri_installer.is_some() {
                cmd.arg("MiriGroup.wixobj");
            }

            if target.contains("windows-gnu") {
                cmd.arg("GccGroup.wixobj");
            }
            // ICE57 wrongly complains about the shortcuts
            cmd.arg("-sice:ICE57");

            let _time = timeit(builder);
            builder.run(&mut cmd);

            if !builder.config.dry_run {
                t!(fs::rename(exe.join(&filename), distdir(builder).join(&filename)));
            }
        }
    }
}

fn add_env(builder: &Builder<'_>, cmd: &mut Command, target: TargetSelection) {
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

    if target.contains("windows-gnu") {
        cmd.env("CFG_MINGW", "1").env("CFG_ABI", "GNU");
    } else {
        cmd.env("CFG_MINGW", "0").env("CFG_ABI", "MSVC");
    }

    if target.contains("x86_64") {
        cmd.env("CFG_PLATFORM", "x64");
    } else {
        cmd.env("CFG_PLATFORM", "x86");
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, Hash, PartialEq, Eq)]
pub struct HashSign;

impl Step for HashSign {
    type Output = ();
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("hash-and-sign")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(HashSign);
    }

    fn run(self, builder: &Builder<'_>) {
        // This gets called by `promote-release`
        // (https://github.com/rust-lang/rust-central-station/tree/master/promote-release).
        let mut cmd = builder.tool_cmd(Tool::BuildManifest);
        if builder.config.dry_run {
            return;
        }
        let sign = builder.config.dist_sign_folder.as_ref().unwrap_or_else(|| {
            panic!("\n\nfailed to specify `dist.sign-folder` in `config.toml`\n\n")
        });
        let addr = builder.config.dist_upload_addr.as_ref().unwrap_or_else(|| {
            panic!("\n\nfailed to specify `dist.upload-addr` in `config.toml`\n\n")
        });
        let pass = if env::var("BUILD_MANIFEST_DISABLE_SIGNING").is_err() {
            let file = builder.config.dist_gpg_password_file.as_ref().unwrap_or_else(|| {
                panic!("\n\nfailed to specify `dist.gpg-password-file` in `config.toml`\n\n")
            });
            t!(fs::read_to_string(&file))
        } else {
            String::new()
        };

        let today = output(Command::new("date").arg("+%Y-%m-%d"));

        cmd.arg(sign);
        cmd.arg(distdir(builder));
        cmd.arg(today.trim());
        cmd.arg(addr);
        cmd.arg(&builder.config.channel);
        cmd.env("BUILD_MANIFEST_LEGACY", "1");

        builder.create_dir(&distdir(builder));

        let mut child = t!(cmd.stdin(Stdio::piped()).spawn());
        t!(child.stdin.take().unwrap().write_all(pass.as_bytes()));
        let status = t!(child.wait());
        assert!(status.success());
    }
}

/// Maybe add libLLVM.so to the given destination lib-dir. It will only have
/// been built if LLVM tools are linked dynamically.
///
/// Note: This function does not yet support Windows, but we also don't support
///       linking LLVM tools dynamically on Windows yet.
fn maybe_install_llvm(builder: &Builder<'_>, target: TargetSelection, dst_libdir: &Path) {
    if !builder.config.llvm_link_shared {
        // We do not need to copy LLVM files into the sysroot if it is not
        // dynamically linked; it is already included into librustc_llvm
        // statically.
        return;
    }

    // On macOS, rustc (and LLVM tools) link to an unversioned libLLVM.dylib
    // instead of libLLVM-11-rust-....dylib, as on linux. It's not entirely
    // clear why this is the case, though. llvm-config will emit the versioned
    // paths and we don't want those in the sysroot (as we're expecting
    // unversioned paths).
    if target.contains("apple-darwin") {
        let src_libdir = builder.llvm_out(target).join("lib");
        let llvm_dylib_path = src_libdir.join("libLLVM.dylib");
        if llvm_dylib_path.exists() {
            builder.install(&llvm_dylib_path, dst_libdir, 0o644);
        }
    } else if let Ok(llvm_config) = crate::native::prebuilt_llvm_config(builder, target) {
        let files = output(Command::new(llvm_config).arg("--libfiles"));
        for file in files.lines() {
            builder.install(Path::new(file), dst_libdir, 0o644);
        }
    }
}

/// Maybe add libLLVM.so to the target lib-dir for linking.
pub fn maybe_install_llvm_target(builder: &Builder<'_>, target: TargetSelection, sysroot: &Path) {
    let dst_libdir = sysroot.join("lib/rustlib").join(&*target.triple).join("lib");
    maybe_install_llvm(builder, target, &dst_libdir);
}

/// Maybe add libLLVM.so to the runtime lib-dir for rustc itself.
pub fn maybe_install_llvm_runtime(builder: &Builder<'_>, target: TargetSelection, sysroot: &Path) {
    let dst_libdir =
        sysroot.join(builder.sysroot_libdir_relative(Compiler { stage: 1, host: target }));
    maybe_install_llvm(builder, target, &dst_libdir);
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LlvmTools {
    pub target: TargetSelection,
}

impl Step for LlvmTools {
    type Output = Option<PathBuf>;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("llvm-tools")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(LlvmTools { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> Option<PathBuf> {
        let target = self.target;
        assert!(builder.config.extended);

        /* run only if llvm-config isn't used */
        if let Some(config) = builder.config.target_config.get(&target) {
            if let Some(ref _s) = config.llvm_config {
                builder.info(&format!("Skipping LlvmTools ({}): external LLVM", target));
                return None;
            }
        }

        builder.info(&format!("Dist LlvmTools ({})", target));
        let _time = timeit(builder);
        let src = builder.src.join("src/llvm-project/llvm");
        let name = pkgname(builder, "llvm-tools");

        let tmp = tmpdir(builder);
        let image = tmp.join("llvm-tools-image");
        drop(fs::remove_dir_all(&image));

        // Prepare the image directory
        let src_bindir = builder.llvm_out(target).join("bin");
        let dst_bindir = image.join("lib/rustlib").join(&*target.triple).join("bin");
        t!(fs::create_dir_all(&dst_bindir));
        for tool in LLVM_TOOLS {
            let exe = src_bindir.join(exe(tool, target));
            builder.install(&exe, &dst_bindir, 0o755);
        }

        // Copy libLLVM.so to the target lib dir as well, so the RPATH like
        // `$ORIGIN/../lib` can find it. It may also be used as a dependency
        // of `rustc-dev` to support the inherited `-lLLVM` when using the
        // compiler libraries.
        maybe_install_llvm_target(builder, target, &image);

        // Prepare the overlay
        let overlay = tmp.join("llvm-tools-overlay");
        drop(fs::remove_dir_all(&overlay));
        builder.create_dir(&overlay);
        builder.install(&src.join("README.txt"), &overlay, 0o644);
        builder.install(&src.join("LICENSE.TXT"), &overlay, 0o644);
        builder.create(&overlay.join("version"), &builder.llvm_tools_vers());

        // Generate the installer tarball
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=llvm-tools-installed.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--component-name=llvm-tools-preview");

        builder.run(&mut cmd);
        Some(distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple)))
    }
}

// Tarball intended for internal consumption to ease rustc/std development.
//
// Should not be considered stable by end users.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RustDev {
    pub target: TargetSelection,
}

impl Step for RustDev {
    type Output = Option<PathBuf>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("rust-dev")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustDev { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> Option<PathBuf> {
        let target = self.target;

        /* run only if llvm-config isn't used */
        if let Some(config) = builder.config.target_config.get(&target) {
            if let Some(ref _s) = config.llvm_config {
                builder.info(&format!("Skipping RustDev ({}): external LLVM", target));
                return None;
            }
        }

        builder.info(&format!("Dist RustDev ({})", target));
        let _time = timeit(builder);
        let src = builder.src.join("src/llvm-project/llvm");
        let name = pkgname(builder, "rust-dev");

        let tmp = tmpdir(builder);
        let image = tmp.join("rust-dev-image");
        drop(fs::remove_dir_all(&image));

        // Prepare the image directory
        let dst_bindir = image.join("bin");
        t!(fs::create_dir_all(&dst_bindir));

        let src_bindir = builder.llvm_out(target).join("bin");
        let install_bin =
            |name| builder.install(&src_bindir.join(exe(name, target)), &dst_bindir, 0o755);
        install_bin("llvm-config");
        install_bin("llvm-ar");
        install_bin("llvm-objdump");
        install_bin("llvm-profdata");
        install_bin("llvm-bcanalyzer");
        install_bin("llvm-cov");
        builder.install(&builder.llvm_filecheck(target), &dst_bindir, 0o755);

        // Copy the include directory as well; needed mostly to build
        // librustc_llvm properly (e.g., llvm-config.h is in here). But also
        // just broadly useful to be able to link against the bundled LLVM.
        builder.cp_r(&builder.llvm_out(target).join("include"), &image.join("include"));

        // Copy libLLVM.so to the target lib dir as well, so the RPATH like
        // `$ORIGIN/../lib` can find it. It may also be used as a dependency
        // of `rustc-dev` to support the inherited `-lLLVM` when using the
        // compiler libraries.
        maybe_install_llvm(builder, target, &image.join("lib"));

        // Prepare the overlay
        let overlay = tmp.join("rust-dev-overlay");
        drop(fs::remove_dir_all(&overlay));
        builder.create_dir(&overlay);
        builder.install(&src.join("README.txt"), &overlay, 0o644);
        builder.install(&src.join("LICENSE.TXT"), &overlay, 0o644);
        builder.create(&overlay.join("version"), &builder.rust_version());

        // Generate the installer tarball
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=rust-dev-installed.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--component-name=rust-dev");

        builder.run(&mut cmd);
        Some(distdir(builder).join(format!("{}-{}.tar.gz", name, target.triple)))
    }
}

/// Tarball containing a prebuilt version of the build-manifest tool, intented to be used by the
/// release process to avoid cloning the monorepo and building stuff.
///
/// Should not be considered stable by end users.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BuildManifest {
    pub target: TargetSelection,
}

impl Step for BuildManifest {
    type Output = PathBuf;
    const DEFAULT: bool = false;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/build-manifest")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(BuildManifest { target: run.target });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let build_manifest = builder.tool_exe(Tool::BuildManifest);

        let name = pkgname(builder, "build-manifest");
        let tmp = tmpdir(builder);

        // Prepare the image.
        let image = tmp.join("build-manifest-image");
        let image_bin = image.join("bin");
        let _ = fs::remove_dir_all(&image);
        t!(fs::create_dir_all(&image_bin));
        builder.install(&build_manifest, &image_bin, 0o755);

        // Prepare the overlay.
        let overlay = tmp.join("build-manifest-overlay");
        let _ = fs::remove_dir_all(&overlay);
        builder.create_dir(&overlay);
        builder.create(&overlay.join("version"), &builder.rust_version());
        for file in &["COPYRIGHT", "LICENSE-APACHE", "LICENSE-MIT", "README.md"] {
            builder.install(&builder.src.join(file), &overlay, 0o644);
        }

        // Create the final tarball.
        let mut cmd = rust_installer(builder);
        cmd.arg("generate")
            .arg("--product-name=Rust")
            .arg("--rel-manifest-dir=rustlib")
            .arg("--success-message=build-manifest installed.")
            .arg("--image-dir")
            .arg(&image)
            .arg("--work-dir")
            .arg(&tmpdir(builder))
            .arg("--output-dir")
            .arg(&distdir(builder))
            .arg("--non-installed-overlay")
            .arg(&overlay)
            .arg(format!("--package-name={}-{}", name, self.target.triple))
            .arg("--legacy-manifest-dirs=rustlib,cargo")
            .arg("--component-name=build-manifest");

        builder.run(&mut cmd);
        distdir(builder).join(format!("{}-{}.tar.gz", name, self.target.triple))
    }
}
