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
    // If we have gcc submodule initialized already, sync it.
    builder.update_existing_submodule("src/gcc");

    // FIXME (GuillaumeGomez): To be done once gccjit has been built in the CI.
    // builder.config.maybe_download_ci_gcc();

    // Initialize the gcc submodule if not initialized already.
    builder.update_submodule("src/gcc");

    let root = "src/gcc";
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

    GccBuildStatus::ShouldBuild(Meta { stamp, out_dir, install_dir, root: root.into() })
}

// FIXME (GuillaumeGomez): When gcc-ci-download option is added, uncomment this code.
// /// This retrieves the GCC sha we *want* to use, according to git history.
// pub(crate) fn detect_gcc_sha(config: &Config, is_git: bool) -> String {
//     let gcc_sha = if is_git {
//         // We proceed in 2 steps. First we get the closest commit that is actually upstream. Then we
//         // walk back further to the last bors merge commit that actually changed GCC. The first
//         // step will fail on CI because only the `auto` branch exists; we just fall back to `HEAD`
//         // in that case.
//         let closest_upstream = get_git_merge_base(&config.git_config(), Some(&config.src))
//             .unwrap_or_else(|_| "HEAD".into());
//         let mut rev_list = config.git();
//         rev_list.args(&[
//             PathBuf::from("rev-list"),
//             format!("--author={}", config.stage0_metadata.config.git_merge_commit_email).into(),
//             "-n1".into(),
//             "--first-parent".into(),
//             closest_upstream.into(),
//             "--".into(),
//             config.src.join("src/gcc"),
//             config.src.join("src/bootstrap/download-ci-gcc-stamp"),
//             // the GCC shared object file is named `gcc-12-rust-{version}-nightly`
//             config.src.join("src/version"),
//         ]);
//         output(&mut rev_list).trim().to_owned()
//     } else if let Some(info) = channel::read_commit_info_file(&config.src) {
//         info.sha.trim().to_owned()
//     } else {
//         "".to_owned()
//     };

//     if gcc_sha.is_empty() {
//         eprintln!("error: could not find commit hash for downloading GCC");
//         eprintln!("HELP: maybe your repository history is too shallow?");
//         eprintln!("HELP: consider disabling `download-ci-gcc`");
//         eprintln!("HELP: or fetch enough history to include one upstream commit");
//         panic!();
//     }

//     gcc_sha
// }

// /// Returns whether the CI-found GCC is currently usable.
// ///
// /// This checks both the build triple platform to confirm we're usable at all,
// /// and then verifies if the current HEAD matches the detected GCC SHA head,
// /// in which case GCC is indicated as not available.
// pub(crate) fn is_ci_gcc_available(config: &Config, asserts: bool) -> bool {
//     let supported_platforms = [
//         // tier 1
//         ("x86_64-unknown-linux-gnu", true),
//     ];

//     if !supported_platforms.contains(&(&*config.build.triple, asserts))
//         && (asserts || !supported_platforms.contains(&(&*config.build.triple, true)))
//     {
//         return false;
//     }

//     if is_ci_gcc_modified(config) {
//         eprintln!("Detected GCC as non-available: running in CI and modified GCC in this change");
//         return false;
//     }

//     true
// }

// /// Returns true if we're running in CI with modified GCC (and thus can't download it)
// pub(crate) fn is_ci_gcc_modified(config: &Config) -> bool {
//     CiEnv::is_ci() && config.rust_info.is_managed_git_subrepository() && {
//         // We assume we have access to git, so it's okay to unconditionally pass
//         // `true` here.
//         let gcc_sha = detect_gcc_sha(config, true);
//         let head_sha = output(config.git().arg("rev-parse").arg("HEAD"));
//         let head_sha = head_sha.trim();
//         gcc_sha == head_sha
//     }
// }

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Gcc {
    pub target: TargetSelection,
}

impl Step for Gcc {
    type Output = bool;

    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/gcc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Gcc { target: run.target });
    }

    /// Compile GCC for `target`.
    fn run(self, builder: &Builder<'_>) -> bool {
        let target = self.target;
        if !target.contains("linux") || !target.contains("x86_64") {
            return false;
        }

        // If GCC has already been built or been downloaded through download-ci-gcc, we avoid
        // building it again.
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

        t!(builder.symlink_file(
            install_dir.join("lib/libgccjit.so"),
            install_dir.join("lib/libgccjit.so.0")
        ));

        t!(stamp.write());

        true
    }
}
