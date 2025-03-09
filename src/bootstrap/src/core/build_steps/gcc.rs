//! Compilation of native dependencies like GCC.
//!
//! Native projects like GCC unfortunately aren't suited just yet for
//! compilation in build scripts that Cargo has. This is because the
//! compilation takes a *very* long time but also because we don't want to
//! compile GCC 3 times as part of a normal bootstrap (we want it cached).
//!
//! GCC and compiler-rt are essentially just wired up to everything else to
//! ensure that they're always in place if needed.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use build_helper::ci::CiEnv;

use crate::Kind;
use crate::core::builder::{Builder, Cargo, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::build_stamp::{BuildStamp, generate_smart_stamp_hash};
use crate::utils::exec::command;
use crate::utils::helpers::{self, t};

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

/// This returns whether we've already previously built GCC.
///
/// It's used to avoid busting caches during x.py check -- if we've already built
/// GCC, it's fine for us to not try to avoid doing so.
pub fn prebuilt_gcc_config(builder: &Builder<'_>, target: TargetSelection) -> GccBuildStatus {
    // Initialize the gcc submodule if not initialized already.
    builder.config.update_submodule("src/gcc");

    let root = builder.src.join("src/gcc");
    let out_dir = builder.gcc_out(target).join("build");
    let install_dir = builder.gcc_out(target).join("install");

    static STAMP_HASH_MEMO: OnceLock<String> = OnceLock::new();
    let smart_stamp_hash = STAMP_HASH_MEMO.get_or_init(|| {
        generate_smart_stamp_hash(
            builder,
            &builder.config.src.join("src/gcc"),
            builder.in_tree_gcc_info.sha().unwrap_or_default(),
        )
    });

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

/// Returns the path to a libgccjit.so file in the install directory of GCC.
fn libgccjit_built_path(install_dir: &Path) -> PathBuf {
    install_dir.join("lib/libgccjit.so")
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Gcc {
    pub target: TargetSelection,
}

#[derive(Clone)]
pub struct GccOutput {
    pub libgccjit: PathBuf,
}

impl Step for Gcc {
    type Output = GccOutput;

    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/gcc").alias("gcc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Gcc { target: run.target });
    }

    /// Compile GCC (specifically `libgccjit`) for `target`.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let target = self.target;

        // If GCC has already been built, we avoid building it again.
        let Meta { stamp, out_dir, install_dir, root } = match prebuilt_gcc_config(builder, target)
        {
            GccBuildStatus::AlreadyBuilt(path) => return GccOutput { libgccjit: path },
            GccBuildStatus::ShouldBuild(m) => m,
        };

        let _guard = builder.msg_unstaged(Kind::Build, "GCC", target);
        t!(stamp.remove());
        let _time = helpers::timeit(builder);
        t!(fs::create_dir_all(&out_dir));
        t!(fs::create_dir_all(&install_dir));

        let libgccjit_path = libgccjit_built_path(&install_dir);
        if builder.config.dry_run() {
            return GccOutput { libgccjit: libgccjit_path };
        }

        // GCC creates files (e.g. symlinks to the downloaded dependencies)
        // in the source directory, which does not work with our CI setup, where we mount
        // source directories as read-only on Linux.
        // Therefore, as a part of the build in CI, we first copy the whole source directory
        // to the build directory, and perform the build from there.
        let src_dir = if CiEnv::is_ci() {
            let src_dir = builder.gcc_out(target).join("src");
            if src_dir.exists() {
                builder.remove_dir(&src_dir);
            }
            builder.create_dir(&src_dir);
            builder.cp_link_r(&root, &src_dir);
            src_dir
        } else {
            root
        };

        command(src_dir.join("contrib/download_prerequisites")).current_dir(&src_dir).run(builder);
        let mut configure_cmd = command(src_dir.join("configure"));
        configure_cmd
            .current_dir(&out_dir)
            // On CI, we compile GCC with Clang.
            // The -Wno-everything flag is needed to make GCC compile with Clang 19.
            // `-g -O2` are the default flags that are otherwise used by Make.
            // FIXME(kobzol): change the flags once we have [gcc] configuration in config.toml.
            .env("CXXFLAGS", "-Wno-everything -g -O2")
            .env("CFLAGS", "-Wno-everything -g -O2")
            .arg("--enable-host-shared")
            .arg("--enable-languages=jit")
            .arg("--enable-checking=release")
            .arg("--disable-bootstrap")
            .arg("--disable-multilib")
            .arg(format!("--prefix={}", install_dir.display()));
        let cc = builder.build.cc(target).display().to_string();
        let cc = builder
            .build
            .config
            .ccache
            .as_ref()
            .map_or_else(|| cc.clone(), |ccache| format!("{ccache} {cc}"));
        configure_cmd.env("CC", cc);

        if let Ok(ref cxx) = builder.build.cxx(target) {
            let cxx = cxx.display().to_string();
            let cxx = builder
                .build
                .config
                .ccache
                .as_ref()
                .map_or_else(|| cxx.clone(), |ccache| format!("{ccache} {cxx}"));
            configure_cmd.env("CXX", cxx);
        }
        configure_cmd.run(builder);

        command("make")
            .current_dir(&out_dir)
            .arg("--silent")
            .arg(format!("-j{}", builder.jobs()))
            .run_capture_stdout(builder);
        command("make")
            .current_dir(&out_dir)
            .arg("--silent")
            .arg("install")
            .run_capture_stdout(builder);

        let lib_alias = install_dir.join("lib/libgccjit.so.0");
        if !lib_alias.exists() {
            t!(builder.symlink_file(&libgccjit_path, lib_alias));
        }

        t!(stamp.write());

        GccOutput { libgccjit: libgccjit_path }
    }
}

/// Configures a Cargo invocation so that it can build the GCC codegen backend.
pub fn add_cg_gcc_cargo_flags(cargo: &mut Cargo, gcc: &GccOutput) {
    // Add the path to libgccjit.so to the linker search paths.
    cargo.rustflag(&format!("-L{}", gcc.libgccjit.parent().unwrap().to_str().unwrap()));
}
