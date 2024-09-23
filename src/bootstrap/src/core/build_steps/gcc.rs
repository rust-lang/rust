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

use build_helper::ci::CiEnv;
use build_helper::git::get_closest_merge_commit;

use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::core::config::{Config, TargetSelection};
use crate::utils::channel;
use crate::utils::exec::command;
use crate::utils::helpers::{self, HashStamp, output, t};
use crate::{Kind, generate_smart_stamp_hash};

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

    builder.config.maybe_download_ci_gcc();

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

/// This retrieves the GCC sha we *want* to use, according to git history.
pub(crate) fn detect_gcc_sha(config: &Config, is_git: bool) -> String {
    let gcc_sha = if is_git {
        get_closest_merge_commit(Some(&config.src), &config.git_config(), &[
            config.src.join("src/gcc"),
            config.src.join("src/bootstrap/download-ci-gcc-stamp"),
            // the GCC shared object file is named `GCC-rust-{version}-nightly`
            config.src.join("src/version"),
        ])
        .unwrap()
    } else if let Some(info) = channel::read_commit_info_file(&config.src) {
        info.sha.trim().to_owned()
    } else {
        "".to_owned()
    };

    if gcc_sha.is_empty() {
        eprintln!("error: could not find commit hash for downloading GCC");
        eprintln!("HELP: maybe your repository history is too shallow?");
        eprintln!("HELP: consider disabling `download-ci-gcc`");
        eprintln!("HELP: or fetch enough history to include one upstream commit");
        panic!();
    }

    gcc_sha
}

/// Returns whether the CI-found GCC is currently usable.
///
/// This checks both the build triple platform to confirm we're usable at all,
/// and then verifies if the current HEAD matches the detected GCC SHA head,
/// in which case GCC is indicated as not available.
pub(crate) fn is_ci_gcc_available(config: &Config, asserts: bool) -> bool {
    // This is currently all tier 1 targets and tier 2 targets with host tools
    // (since others may not have CI artifacts)
    // https://doc.rust-lang.org/rustc/platform-support.html#tier-1
    let supported_platforms = [
        // tier 1
        ("aarch64-unknown-linux-gnu", false),
        ("aarch64-apple-darwin", false),
        ("i686-pc-windows-gnu", false),
        ("i686-pc-windows-msvc", false),
        ("i686-unknown-linux-gnu", false),
        ("x86_64-unknown-linux-gnu", true),
        ("x86_64-apple-darwin", true),
        ("x86_64-pc-windows-gnu", true),
        ("x86_64-pc-windows-msvc", true),
        // tier 2 with host tools
        ("aarch64-pc-windows-msvc", false),
        ("aarch64-unknown-linux-musl", false),
        ("arm-unknown-linux-gnueabi", false),
        ("arm-unknown-linux-gnueabihf", false),
        ("armv7-unknown-linux-gnueabihf", false),
        ("loongarch64-unknown-linux-gnu", false),
        ("loongarch64-unknown-linux-musl", false),
        ("mips-unknown-linux-gnu", false),
        ("mips64-unknown-linux-gnuabi64", false),
        ("mips64el-unknown-linux-gnuabi64", false),
        ("mipsel-unknown-linux-gnu", false),
        ("powerpc-unknown-linux-gnu", false),
        ("powerpc64-unknown-linux-gnu", false),
        ("powerpc64le-unknown-linux-gnu", false),
        ("riscv64gc-unknown-linux-gnu", false),
        ("s390x-unknown-linux-gnu", false),
        ("x86_64-unknown-freebsd", false),
        ("x86_64-unknown-illumos", false),
        ("x86_64-unknown-linux-musl", false),
        ("x86_64-unknown-netbsd", false),
    ];

    if !supported_platforms.contains(&(&*config.build.triple, asserts))
        && (asserts || !supported_platforms.contains(&(&*config.build.triple, true)))
    {
        return false;
    }

    if is_ci_gcc_modified(config) {
        eprintln!("Detected GCC as non-available: running in CI and modified GCC in this change");
        return false;
    }

    true
}

/// Returns true if we're running in CI with modified LLVM (and thus can't download it)
pub(crate) fn is_ci_gcc_modified(config: &Config) -> bool {
    CiEnv::is_ci() && config.rust_info.is_managed_git_subrepository() && {
        // We assume we have access to git, so it's okay to unconditionally pass
        // `true` here.
        let gcc_sha = detect_gcc_sha(config, true);
        let head_sha =
            output(helpers::git(Some(&config.src)).arg("rev-parse").arg("HEAD").as_command_mut());
        let head_sha = head_sha.trim();
        gcc_sha == head_sha
    }
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
