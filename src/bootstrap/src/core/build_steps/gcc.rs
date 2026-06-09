//! Compilation of native dependencies like GCC.
//!
//! Native projects like GCC unfortunately aren't suited just yet for
//! compilation in build scripts that Cargo has. This is because the
//! compilation takes a *very* long time but also because we don't want to
//! compile GCC 3 times as part of a normal bootstrap (we want it cached).
//!
//! GCC and compiler-rt are essentially just wired up to everything else to
//! ensure that they're always in place if needed.

use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::core::builder::{Builder, Cargo, Kind, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::build_stamp::{BuildStamp, generate_smart_stamp_hash};
use crate::utils::exec::command;
use crate::utils::helpers::{self, t};

/// GCC cannot cross-compile from a single binary to multiple targets.
/// So we need to have a separate GCC dylib for each (host, target) pair.
/// We represent this explicitly using this struct.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct GccTargetPair {
    /// Target on which the libgccjit.so dylib will be executed.
    host: TargetSelection,
    /// Target for which the libgccjit.so dylib will generate assembly.
    target: TargetSelection,
}

impl GccTargetPair {
    /// Create a target pair for a GCC that will run on `target` and generate assembly for `target`.
    pub fn for_native_build(target: TargetSelection) -> Self {
        Self { host: target, target }
    }

    /// Create a target pair for a GCC that will run on `host` and generate assembly for `target`.
    /// This may be cross-compilation if `host != target`.
    pub fn for_target_pair(host: TargetSelection, target: TargetSelection) -> Self {
        Self { host, target }
    }

    pub fn host(&self) -> TargetSelection {
        self.host
    }

    pub fn target(&self) -> TargetSelection {
        self.target
    }
}

impl Display for GccTargetPair {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.host, self.target)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Gcc {
    pub target_pair: GccTargetPair,
}

#[derive(Clone)]
pub struct GccOutput {
    /// Path to a built or downloaded libgccjit.
    libgccjit: PathBuf,
}

impl GccOutput {
    pub fn libgccjit(&self) -> &Path {
        &self.libgccjit
    }
}

impl Step for Gcc {
    type Output = GccOutput;

    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/gcc").alias("gcc")
    }

    fn make_run(run: RunConfig<'_>) {
        // By default, we build libgccjit that can do native compilation (no cross-compilation)
        // on a given target.
        run.builder
            .ensure(Gcc { target_pair: GccTargetPair { host: run.target, target: run.target } });
    }

    /// Compile GCC (specifically `libgccjit`) for `target`.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let target_pair = self.target_pair;

        // If GCC has already been built, we avoid building it again.
        let metadata = match get_gcc_build_status(builder, target_pair) {
            GccBuildStatus::AlreadyBuilt(path) => return GccOutput { libgccjit: path },
            GccBuildStatus::ShouldBuild(m) => m,
        };

        let action = Kind::Build.description();
        let msg = format!("{action} GCC for {target_pair}");
        let _guard = builder.group(&msg);
        t!(metadata.stamp.remove());
        let _time = helpers::timeit(builder);

        let libgccjit_path = libgccjit_built_path(&metadata.install_dir);
        if builder.config.dry_run() {
            return GccOutput { libgccjit: libgccjit_path };
        }

        build_gcc(&metadata, builder, target_pair);

        t!(metadata.stamp.write());

        GccOutput { libgccjit: libgccjit_path }
    }
}

pub struct Meta {
    stamp: BuildStamp,
    out_dir: PathBuf,
    install_dir: PathBuf,
    root: PathBuf,
}

pub enum GccBuildStatus {
    /// libgccjit is already built at this path
    AlreadyBuilt(PathBuf),
    ShouldBuild(Meta),
}

/// Tries to download GCC from CI if it is enabled and GCC artifacts
/// are available for the given target.
/// Returns a path to the libgccjit.so file.
#[cfg(not(test))]
fn try_download_gcc(builder: &Builder<'_>, target_pair: GccTargetPair) -> Option<PathBuf> {
    use build_helper::git::PathFreshness;

    // Try to download GCC from CI if configured and available
    if !matches!(builder.config.gcc_ci_mode, crate::core::config::GccCiMode::DownloadFromCi) {
        return None;
    }

    // We currently do not support downloading CI GCC if the host/target pair doesn't match.
    if target_pair.host != target_pair.target {
        eprintln!(
            "GCC CI download is not available when the host ({}) does not equal the compilation target ({}).",
            target_pair.host, target_pair.target
        );
        return None;
    }

    if target_pair.host != "x86_64-unknown-linux-gnu" {
        eprintln!(
            "GCC CI download is only available for the `x86_64-unknown-linux-gnu` host/target"
        );
        return None;
    }
    let source = detect_gcc_freshness(
        &builder.config,
        builder.config.rust_info.is_managed_git_subrepository(),
    );
    builder.do_if_verbose(|| {
        eprintln!("GCC freshness: {source:?}");
    });
    match source {
        PathFreshness::LastModifiedUpstream { upstream } => {
            // Download from upstream CI
            let root = ci_gcc_root(&builder.config, target_pair.target);
            let gcc_stamp = BuildStamp::new(&root).with_prefix("gcc").add_stamp(&upstream);
            if !gcc_stamp.is_up_to_date() && !builder.config.dry_run() {
                builder.config.download_ci_gcc(&upstream, &root);
                t!(gcc_stamp.write());
            }

            let libgccjit = root.join("lib").join("libgccjit.so");
            Some(libgccjit)
        }
        PathFreshness::HasLocalModifications { .. } => {
            // We have local modifications, rebuild GCC.
            eprintln!("Found local GCC modifications, GCC will *not* be downloaded");
            None
        }
        PathFreshness::MissingUpstream => {
            eprintln!("error: could not find commit hash for downloading GCC");
            eprintln!("HELP: maybe your repository history is too shallow?");
            eprintln!("HELP: consider disabling `download-ci-gcc`");
            eprintln!("HELP: or fetch enough history to include one upstream commit");
            None
        }
    }
}

#[cfg(test)]
fn try_download_gcc(_builder: &Builder<'_>, _target_pair: GccTargetPair) -> Option<PathBuf> {
    None
}

/// This returns information about whether GCC should be built or if it's already built.
/// It transparently handles downloading GCC from CI if needed.
///
/// It's used to avoid busting caches during x.py check -- if we've already built
/// GCC, it's fine for us to not try to avoid doing so.
pub fn get_gcc_build_status(builder: &Builder<'_>, target_pair: GccTargetPair) -> GccBuildStatus {
    // Prefer taking externally provided prebuilt libgccjit dylib
    if let Some(dir) = &builder.config.libgccjit_libs_dir {
        // The dir structure should be <root>/<host>/<target>/libgccjit.so
        let host_dir = dir.join(target_pair.host);
        let path = host_dir.join(target_pair.target).join("libgccjit.so");
        if path.exists() {
            return GccBuildStatus::AlreadyBuilt(path);
        } else {
            builder.info(&format!(
                "libgccjit.so for `{target_pair}` was not found at `{}`",
                path.display()
            ));

            if target_pair.host != target_pair.target || target_pair.host != builder.host_target {
                eprintln!(
                    "info: libgccjit.so for `{target_pair}` was not found at `{}`",
                    path.display()
                );
                eprintln!("error: we do not support downloading or building a GCC cross-compiler");
                std::process::exit(1);
            }
        }
    }

    // If not available, try to download from CI
    if let Some(path) = try_download_gcc(builder, target_pair) {
        return GccBuildStatus::AlreadyBuilt(path);
    }

    // If not available, try to build (or use already built libgccjit from disk)
    static STAMP_HASH_MEMO: OnceLock<String> = OnceLock::new();
    let smart_stamp_hash = STAMP_HASH_MEMO.get_or_init(|| {
        generate_smart_stamp_hash(
            builder,
            &builder.config.src.join("src/gcc"),
            builder.in_tree_gcc_info.sha().unwrap_or_default(),
        )
    });

    // Initialize the gcc submodule if not initialized already.
    builder.config.update_submodule("src/gcc");

    let root = builder.src.join("src/gcc");
    let out_dir = gcc_out(builder, target_pair).join("build");
    let install_dir = gcc_out(builder, target_pair).join("install");

    let stamp = BuildStamp::new(&out_dir).with_prefix("gcc").add_stamp(smart_stamp_hash);

    if stamp.is_up_to_date() {
        if stamp.stamp().is_empty() {
            builder.info(
                "Could not determine the GCC submodule commit hash. \
                     Assuming that an GCC rebuild is not necessary.",
            );
            builder.info(&format!(
                "To force GCC to rebuild, remove the file `{}`",
                stamp.path().display()
            ));
        }
        let path = libgccjit_built_path(&install_dir);
        if path.is_file() {
            return GccBuildStatus::AlreadyBuilt(path);
        } else {
            builder.info(&format!(
                "GCC stamp is up-to-date, but the libgccjit.so file was not found at `{}`",
                path.display(),
            ));
        }
    }

    GccBuildStatus::ShouldBuild(Meta { stamp, out_dir, install_dir, root })
}

fn gcc_out(builder: &Builder<'_>, pair: GccTargetPair) -> PathBuf {
    builder.out.join(pair.host).join("gcc").join(pair.target)
}

/// Returns the path to a libgccjit.so file in the install directory of GCC.
fn libgccjit_built_path(install_dir: &Path) -> PathBuf {
    install_dir.join("lib/libgccjit.so")
}

fn build_gcc(metadata: &Meta, builder: &Builder<'_>, target_pair: GccTargetPair) {
    // Target on which libgccjit.so will be executed. Here we will generate a dylib with
    // instructions for that target.
    let host = target_pair.host;
    if builder.build.cc_tool(host).is_like_clang() || builder.build.cxx_tool(host).is_like_clang() {
        panic!(
            "Attempting to build GCC using Clang, which is known to misbehave. Please use GCC as the host C/C++ compiler. "
        );
    }

    let Meta { stamp: _, out_dir, install_dir, root } = metadata;

    t!(fs::create_dir_all(out_dir));
    t!(fs::create_dir_all(install_dir));

    // GCC creates files (e.g. symlinks to the downloaded dependencies)
    // in the source directory, which does not work with our CI/Docker setup, where we mount
    // source directories as read-only on Linux.
    // And in general, we shouldn't be modifying the source directories if possible, even for local
    // builds.
    // Therefore, we first copy the whole source directory to the build directory, and perform the
    // build from there.
    let src_dir = gcc_out(builder, target_pair).join("src");
    if src_dir.exists() {
        builder.remove_dir(&src_dir);
    }
    builder.create_dir(&src_dir);
    builder.cp_link_r(root, &src_dir);

    command(src_dir.join("contrib/download_prerequisites")).current_dir(&src_dir).run(builder);
    let mut configure_cmd = command(src_dir.join("configure"));
    configure_cmd
        .current_dir(out_dir)
        .arg("--enable-host-shared")
        .arg("--enable-languages=c,jit,lto")
        .arg("--enable-checking=release")
        .arg("--disable-bootstrap")
        .arg("--disable-multilib")
        .arg("--with-bugurl=https://github.com/rust-lang/gcc/")
        .arg(format!("--prefix={}", install_dir.display()));

    let cc = builder.build.cc(host).display().to_string();
    let cc = builder
        .build
        .config
        .ccache
        .as_ref()
        .map_or_else(|| cc.clone(), |ccache| format!("{ccache} {cc}"));
    configure_cmd.env("CC", cc);

    if let Ok(ref cxx) = builder.build.cxx(host) {
        let cxx = cxx.display().to_string();
        let cxx = builder
            .build
            .config
            .ccache
            .as_ref()
            .map_or_else(|| cxx.clone(), |ccache| format!("{ccache} {cxx}"));
        configure_cmd.env("CXX", cxx);
    }
    // Disable debuginfo to reduce size of libgccjit.so 10x
    configure_cmd.env("CXXFLAGS", "-O2 -g0");
    configure_cmd.env("CFLAGS", "-O2 -g0");
    configure_cmd.run(builder);

    command("make")
        .current_dir(out_dir)
        .arg("--silent")
        .arg(format!("-j{}", builder.jobs()))
        .run_capture_stdout(builder);
    command("make").current_dir(out_dir).arg("--silent").arg("install").run_capture_stdout(builder);
}

/// Configures a Cargo invocation so that it can build the GCC codegen backend.
pub fn add_cg_gcc_cargo_flags(cargo: &mut Cargo, gcc: &GccOutput) {
    // Add the path to libgccjit.so to the linker search paths.
    cargo.rustflag(&format!("-L{}", gcc.libgccjit.parent().unwrap().to_str().unwrap()));
}

/// The absolute path to the downloaded GCC artifacts.
#[cfg(not(test))]
fn ci_gcc_root(config: &crate::Config, target: TargetSelection) -> PathBuf {
    config.out.join(target).join("ci-gcc")
}

/// Detect whether GCC sources have been modified locally or not.
#[cfg(not(test))]
fn detect_gcc_freshness(config: &crate::Config, is_git: bool) -> build_helper::git::PathFreshness {
    use build_helper::git::PathFreshness;

    if is_git {
        config.check_path_modifications(&["src/gcc", "src/bootstrap/download-ci-gcc-stamp"])
    } else if let Some(info) = crate::utils::channel::read_commit_info_file(&config.src) {
        PathFreshness::LastModifiedUpstream { upstream: info.sha.trim().to_owned() }
    } else {
        PathFreshness::MissingUpstream
    }
}
