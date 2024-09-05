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
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::exec::command;
use crate::utils::helpers::{self, t, HashStamp};
use crate::{generate_smart_stamp_hash, Kind};

pub struct Meta {
    stamp: HashStamp,
    out_dir: PathBuf,
    install_dir: PathBuf,
    root: PathBuf,
}

pub enum GccBuildStatus {
    AlreadyBuilt,
    ShouldBuild(Meta),
}

/// This returns whether we've already previously built GCC.
///
/// It's used to avoid busting caches during x.py check -- if we've already built
/// GCC, it's fine for us to not try to avoid doing so.
pub fn prebuilt_gcc_config(builder: &Builder<'_>, target: TargetSelection) -> GccBuildStatus {
    // Initialize the gcc submodule if not initialized already.
    builder.config.update_submodule("src/gcc");

    // FIXME (GuillaumeGomez): To be done once gccjit has been built in the CI.
    // builder.config.maybe_download_ci_gcc();

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

    let stamp = out_dir.join("gcc-finished-building");
    let stamp = HashStamp::new(stamp, Some(smart_stamp_hash));

    if stamp.is_done() {
        if stamp.hash.is_none() {
            builder.info(
                "Could not determine the GCC submodule commit hash. \
                     Assuming that an GCC rebuild is not necessary.",
            );
            builder.info(&format!(
                "To force GCC to rebuild, remove the file `{}`",
                stamp.path.display()
            ));
        }
        return GccBuildStatus::AlreadyBuilt;
    }

    GccBuildStatus::ShouldBuild(Meta { stamp, out_dir, install_dir, root })
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Gcc {
    pub target: TargetSelection,
}

impl Step for Gcc {
    type Output = bool;

    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/gcc").alias("gcc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Gcc { target: run.target });
    }

    /// Compile GCC for `target`.
    fn run(self, builder: &Builder<'_>) -> bool {
        let target = self.target;

        // If GCC has already been built, we avoid building it again.
        let Meta { stamp, out_dir, install_dir, root } = match prebuilt_gcc_config(builder, target)
        {
            GccBuildStatus::AlreadyBuilt => return true,
            GccBuildStatus::ShouldBuild(m) => m,
        };

        let _guard = builder.msg_unstaged(Kind::Build, "GCC", target);
        t!(stamp.remove());
        let _time = helpers::timeit(builder);
        t!(fs::create_dir_all(&out_dir));

        if builder.config.dry_run() {
            return true;
        }

        command(root.join("contrib/download_prerequisites")).current_dir(&root).run(builder);
        command(root.join("configure"))
            .current_dir(&out_dir)
            .arg("--enable-host-shared")
            .arg("--enable-languages=jit")
            .arg("--enable-checking=release")
            .arg("--disable-bootstrap")
            .arg("--disable-multilib")
            .arg(format!("--prefix={}", install_dir.display()))
            .run(builder);
        command("make").current_dir(&out_dir).arg(format!("-j{}", builder.jobs())).run(builder);
        command("make").current_dir(&out_dir).arg("install").run(builder);

        let lib_alias = install_dir.join("lib/libgccjit.so.0");
        if !lib_alias.exists() {
            t!(builder.symlink_file(install_dir.join("lib/libgccjit.so"), lib_alias,));
        }

        t!(stamp.write());

        true
    }
}
