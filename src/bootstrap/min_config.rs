use core::fmt;
use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde::Deserialize;

use crate::{
    cache::{Interned, INTERNER},
    t,
    util::output,
};

/// The bare minimum config, suitable for `bootstrap-shim`, but sharing code with the main `bootstrap` binary.
#[derive(Default, Clone)]
pub struct MinimalConfig {
    // Needed so we know where to store the unpacked bootstrap binary.
    pub build: TargetSelection,
    // Needed so we know where to load `src/stage0.json`
    pub src: PathBuf,
    // Needed so we know where to store the cache.
    pub out: PathBuf,
    pub patch_binaries_for_nix: bool,
    // Needed to know which commit to download.
    pub stage0_metadata: Stage0Metadata,

    // This isn't currently used, but will eventually let people configure whether to download or build bootstrap.
    pub config: Option<PathBuf>,
    // Not currently used in the shim.
    pub verbose: usize,
    // Not currently used in the shim.
    pub dry_run: DryRun,
}

#[derive(Default, Deserialize, Clone)]
pub struct Stage0Metadata {
    pub compiler: CompilerMetadata,
    pub config: Stage0Config,
    pub checksums_sha256: HashMap<String, String>,
    pub rustfmt: Option<RustfmtMetadata>,
}
#[derive(Clone, Default, Deserialize)]
pub struct CompilerMetadata {
    pub date: String,
    pub version: String,
}
#[derive(Default, Deserialize, Clone)]
pub struct Stage0Config {
    pub dist_server: String,
    pub artifacts_server: String,
    pub artifacts_with_llvm_assertions_server: String,
    pub git_merge_commit_email: String,
    pub nightly_branch: String,
}

#[derive(Default, Deserialize, Clone)]
pub struct RustfmtMetadata {
    pub date: String,
    pub version: String,
}

impl MinimalConfig {
    fn default_opts() -> Self {
        let dry_run = DryRun::default();
        let config = None;
        let verbose = 0;
        let patch_binaries_for_nix = false;
        let stage0_metadata = Stage0Metadata::default();

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // Undo `src/bootstrap`
        let src = manifest_dir.parent().unwrap().parent().unwrap().to_owned();
        let out = PathBuf::from("build");

        // set by build.rs
        let build = TargetSelection::from_user(&env!("BUILD_TRIPLE"));

        MinimalConfig {
            build,
            src,
            out,
            config,
            dry_run,
            verbose,
            patch_binaries_for_nix,
            stage0_metadata,
        }
    }

    pub fn parse(config_flag: Option<PathBuf>) -> MinimalConfig {
        let mut config = Self::default_opts();

        if let Some(src) = src() {
            config.src = src;
        }

        set_config_output_dir(&mut config.out);

        let toml: TomlConfig =
            set_and_return_toml_config(config.src.clone(), config_flag, &mut config.config);

        if let Some(build) = toml.build.unwrap_or_default().build {
            config.build = TargetSelection::from_user(&build);
        }

        if config.dry_run() {
            let dir = config.out.join("tmp-dry-run");
            t!(fs::create_dir_all(&dir));
            config.out = dir;
        }
        // NOTE: Bootstrap spawns various commands with different working directories.
        // To avoid writing to random places on the file system, `config.out` needs to be an absolute path.
        else if !config.out.is_absolute() {
            // `canonicalize` requires the path to already exist. Use our vendored copy of `absolute` instead.
            config.out = crate::util::absolute(&config.out);
        }

        config.stage0_metadata = deserialize_stage0_metadata(&config.src);

        config
    }
}

impl MinimalConfig {
    pub fn verbose(&self, msg: &str) {
        if self.verbose > 0 {
            println!("{}", msg);
        }
    }

    pub(crate) fn dry_run(&self) -> bool {
        match self.dry_run {
            DryRun::Disabled => false,
            DryRun::SelfCheck | DryRun::UserSelected => true,
        }
    }

    /// A git invocation which runs inside the source directory.
    ///
    /// Use this rather than `Command::new("git")` in order to support out-of-tree builds.
    pub(crate) fn git(&self) -> Command {
        let mut git = Command::new("git");
        git.current_dir(&self.src);
        git
    }

    /// Returns the last commit in which any of `modified_paths` were changed,
    /// or `None` if there are untracked changes in the working directory and `if_unchanged` is true.
    pub fn last_modified_commit(
        &self,
        modified_paths: &[&str],
        option_name: &str,
        if_unchanged: bool,
    ) -> Option<String> {
        // Handle running from a directory other than the top level
        let top_level = output(self.git().args(&["rev-parse", "--show-toplevel"]));
        let top_level = top_level.trim_end();

        // Look for a version to compare to based on the current commit.
        // Only commits merged by bors will have CI artifacts.
        let merge_base = output(
            self.git()
                .arg("rev-list")
                .arg(format!("--author={}", self.stage0_metadata.config.git_merge_commit_email))
                .args(&["-n1", "--first-parent", "HEAD"]),
        );
        let commit = merge_base.trim_end();
        if commit.is_empty() {
            println!("error: could not find commit hash for downloading components from CI");
            println!("help: maybe your repository history is too shallow?");
            println!("help: consider disabling `{option_name}`");
            println!("help: or fetch enough history to include one upstream commit");
            crate::detail_exit(1);
        }

        // Warn if there were changes to the compiler or standard library since the ancestor commit.
        let mut git = self.git();
        git.args(&["diff-index", "--quiet", &commit, "--"]);

        for path in modified_paths {
            git.arg(format!("{top_level}/{path}"));
        }

        let has_changes = !t!(git.status()).success();
        if has_changes {
            if if_unchanged {
                if self.verbose > 0 {
                    println!(
                        "warning: saw changes to one of {modified_paths:?} since {commit}; \
                            ignoring `{option_name}`"
                    );
                }
                return None;
            }
            println!(
                "warning: `{option_name}` is enabled, but there are changes to one of {modified_paths:?}"
            );
        }

        Some(commit.to_string())
    }
}

#[cfg(test)]
/// Shared helper function to be used in `MinimalConfig::parse` and `bootstrap::config::Config::parse`
pub(crate) fn get_toml<T: Deserialize<'static> + Default>(_file: &Path) -> T {
    T::default()
}
#[cfg(not(test))]
/// Shared helper function to be used in `MinimalConfig::parse` and `bootstrap::config::Config::parse`
pub(crate) fn get_toml<T: Deserialize<'static> + Default>(file: &Path) -> T {
    let contents =
        t!(fs::read_to_string(file), format!("config file {} not found", file.display()));
    // Deserialize to Value and then TomlConfig to prevent the Deserialize impl of
    // TomlConfig and sub types to be monomorphized 5x by toml.
    match toml::from_str(&contents).and_then(|table: toml::Value| T::deserialize(table)) {
        Ok(table) => table,
        Err(err) => {
            eprintln!("failed to parse TOML configuration '{}': {}", file.display(), err);
            crate::detail_exit(2);
        }
    }
}

/// Shared helper function to be used in `MinimalConfig::parse` and `bootstrap::config::Config::parse`
///
/// Use the build directory of the original x.py invocation, so that we can set `initial_rustc` properly.
#[allow(unused_variables)]
pub(crate) fn set_config_output_dir(output_path: &mut PathBuf) {
    #[cfg(test)]
    {
        *output_path = Path::new(
            &env::var_os("CARGO_TARGET_DIR").expect("cargo test directly is not supported"),
        )
        .parent()
        .unwrap()
        .to_path_buf();
    }
}

/// Shared helper function to be used in `MinimalConfig::parse` and `bootstrap::config::Config::parse`
pub(crate) fn set_and_return_toml_config<T: Deserialize<'static> + Default>(
    src: PathBuf,
    config_flag: Option<PathBuf>,
    cfg_path: &mut Option<PathBuf>,
) -> T {
    /// Read from `--config`, then `RUST_BOOTSTRAP_CONFIG`, then `./config.toml`, then `config.toml` in the root directory.
    ///
    /// Give a hard error if `--config` or `RUST_BOOTSTRAP_CONFIG` are set to a missing path,
    /// but not if `config.toml` hasn't been created.
    fn config_path(src: &PathBuf, config_flag: Option<PathBuf>) -> Option<PathBuf> {
        let toml_path =
            config_flag.or_else(|| env::var_os("RUST_BOOTSTRAP_CONFIG").map(PathBuf::from));
        let using_default_path = toml_path.is_none();
        let mut toml_path = toml_path.unwrap_or_else(|| PathBuf::from("config.toml"));
        if using_default_path && !toml_path.exists() {
            toml_path = src.join(toml_path);
        }

        if !using_default_path || toml_path.exists() { Some(toml_path) } else { None }
    }

    if let Some(toml_path) = config_path(&src, config_flag) {
        *cfg_path = Some(toml_path.clone());
        get_toml(&toml_path)
    } else {
        *cfg_path = None;
        T::default()
    }
}

/// Shared helper function to be used in `MinimalConfig::parse` and `bootstrap::config::Config::parse`
pub(crate) fn deserialize_stage0_metadata(stage0_metadata_path: &PathBuf) -> Stage0Metadata {
    let stage0_json = t!(std::fs::read(stage0_metadata_path.join("src").join("stage0.json")));
    t!(serde_json::from_slice::<Stage0Metadata>(&stage0_json))
}

fn src() -> Option<PathBuf> {
    // Infer the source directory. This is non-trivial because we want to support a downloaded bootstrap binary,
    // running on a completely machine from where it was compiled.
    let mut cmd = Command::new("git");
    // NOTE: we cannot support running from outside the repository because the only path we have available
    // is set at compile time, which can be wrong if bootstrap was downloaded from source.
    // We still support running outside the repository if we find we aren't in a git directory.
    cmd.arg("rev-parse").arg("--show-toplevel");
    // Discard stderr because we expect this to fail when building from a tarball.
    let output = cmd
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .and_then(|output| if output.status.success() { Some(output) } else { None });
    if let Some(output) = output {
        let git_root = String::from_utf8(output.stdout).unwrap();
        // We need to canonicalize this path to make sure it uses backslashes instead of forward slashes.
        let git_root = PathBuf::from(git_root.trim()).canonicalize().unwrap();
        let s = git_root.to_str().unwrap();

        // Bootstrap is quite bad at handling /? in front of paths
        let src = match s.strip_prefix("\\\\?\\") {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(git_root),
        };
        // If this doesn't have at least `stage0.json`, we guessed wrong. This can happen when,
        // for example, the build directory is inside of another unrelated git directory.
        // In that case keep the original `CARGO_MANIFEST_DIR` handling.
        //
        // NOTE: this implies that downloadable bootstrap isn't supported when the build directory is outside
        // the source directory. We could fix that by setting a variable from all three of python, ./x, and x.ps1.
        if src.join("src").join("stage0.json").exists() { Some(src) } else { None }
    } else {
        // We're building from a tarball, not git sources.
        // We don't support pre-downloaded bootstrap in this case.
        None
    }
}

#[derive(Clone, Default)]
pub enum DryRun {
    /// This isn't a dry run.
    #[default]
    Disabled,
    /// This is a dry run enabled by bootstrap itself, so it can verify that no work is done.
    SelfCheck,
    /// This is a dry run enabled by the `--dry-run` flag.
    UserSelected,
}

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetSelection {
    pub triple: Interned<String>,
    file: Option<Interned<String>>,
}

impl TargetSelection {
    pub fn from_user(selection: &str) -> Self {
        let path = Path::new(selection);

        let (triple, file) = if path.exists() {
            let triple = path
                .file_stem()
                .expect("Target specification file has no file stem")
                .to_str()
                .expect("Target specification file stem is not UTF-8");

            (triple, Some(selection))
        } else {
            (selection, None)
        };

        let triple = INTERNER.intern_str(triple);
        let file = file.map(|f| INTERNER.intern_str(f));

        Self { triple, file }
    }

    pub fn rustc_target_arg(&self) -> &str {
        self.file.as_ref().unwrap_or(&self.triple)
    }

    pub fn contains(&self, needle: &str) -> bool {
        self.triple.contains(needle)
    }

    pub fn starts_with(&self, needle: &str) -> bool {
        self.triple.starts_with(needle)
    }

    pub fn ends_with(&self, needle: &str) -> bool {
        self.triple.ends_with(needle)
    }
}

impl fmt::Display for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.triple)?;
        if let Some(file) = self.file {
            write!(f, "({})", file)?;
        }
        Ok(())
    }
}

impl fmt::Debug for TargetSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl PartialEq<&str> for TargetSelection {
    fn eq(&self, other: &&str) -> bool {
        self.triple == *other
    }
}

#[derive(Deserialize, Default)]
struct TomlConfig {
    build: Option<Build>,
}

/// TOML representation of various global build decisions.
#[derive(Deserialize, Default)]
struct Build {
    build: Option<String>,
}
