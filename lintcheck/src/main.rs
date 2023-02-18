// Run clippy on a fixed set of crates and collect the warnings.
// This helps observing the impact clippy changes have on a set of real-world code (and not just our
// testsuite).
//
// When a new lint is introduced, we can search the results for new warnings and check for false
// positives.

#![allow(clippy::collapsible_else_if)]

mod config;
mod driver;
mod recursive;

use crate::config::LintcheckConfig;
use crate::recursive::LintcheckServer;

use std::collections::{HashMap, HashSet};
use std::env;
use std::env::consts::EXE_SUFFIX;
use std::fmt::{self, Write as _};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use cargo_metadata::diagnostic::{Diagnostic, DiagnosticLevel};
use cargo_metadata::Message;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use walkdir::{DirEntry, WalkDir};

const LINTCHECK_DOWNLOADS: &str = "target/lintcheck/downloads";
const LINTCHECK_SOURCES: &str = "target/lintcheck/sources";

/// List of sources to check, loaded from a .toml file
#[derive(Debug, Serialize, Deserialize)]
struct SourceList {
    crates: HashMap<String, TomlCrate>,
    #[serde(default)]
    recursive: RecursiveOptions,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct RecursiveOptions {
    ignore: HashSet<String>,
}

/// A crate source stored inside the .toml
/// will be translated into on one of the `CrateSource` variants
#[derive(Debug, Serialize, Deserialize)]
struct TomlCrate {
    name: String,
    versions: Option<Vec<String>>,
    git_url: Option<String>,
    git_hash: Option<String>,
    path: Option<String>,
    options: Option<Vec<String>>,
}

/// Represents an archive we download from crates.io, or a git repo, or a local repo/folder
/// Once processed (downloaded/extracted/cloned/copied...), this will be translated into a `Crate`
#[derive(Debug, Serialize, Deserialize, Eq, Hash, PartialEq, Ord, PartialOrd)]
enum CrateSource {
    CratesIo {
        name: String,
        version: String,
        options: Option<Vec<String>>,
    },
    Git {
        name: String,
        url: String,
        commit: String,
        options: Option<Vec<String>>,
    },
    Path {
        name: String,
        path: PathBuf,
        options: Option<Vec<String>>,
    },
}

/// Represents the actual source code of a crate that we ran "cargo clippy" on
#[derive(Debug)]
struct Crate {
    version: String,
    name: String,
    // path to the extracted sources that clippy can check
    path: PathBuf,
    options: Option<Vec<String>>,
}

/// A single warning that clippy issued while checking a `Crate`
#[derive(Debug)]
struct ClippyWarning {
    crate_name: String,
    file: String,
    line: usize,
    column: usize,
    lint_type: String,
    message: String,
    is_ice: bool,
}

#[allow(unused)]
impl ClippyWarning {
    fn new(diag: Diagnostic, crate_name: &str, crate_version: &str) -> Option<Self> {
        let lint_type = diag.code?.code;
        if !(lint_type.contains("clippy") || diag.message.contains("clippy"))
            || diag.message.contains("could not read cargo metadata")
        {
            return None;
        }

        let span = diag.spans.into_iter().find(|span| span.is_primary)?;

        let file = if let Ok(stripped) = Path::new(&span.file_name).strip_prefix(env!("CARGO_HOME")) {
            format!("$CARGO_HOME/{}", stripped.display())
        } else {
            format!(
                "target/lintcheck/sources/{crate_name}-{crate_version}/{}",
                span.file_name
            )
        };

        Some(Self {
            crate_name: crate_name.to_owned(),
            file,
            line: span.line_start,
            column: span.column_start,
            lint_type,
            message: diag.message,
            is_ice: diag.level == DiagnosticLevel::Ice,
        })
    }

    fn to_output(&self, markdown: bool) -> String {
        let file_with_pos = format!("{}:{}:{}", &self.file, &self.line, &self.column);
        if markdown {
            let mut file = self.file.clone();
            if !file.starts_with('$') {
                file.insert_str(0, "../");
            }

            let mut output = String::from("| ");
            let _: fmt::Result = write!(output, "[`{file_with_pos}`]({file}#L{})", self.line);
            let _: fmt::Result = write!(output, r#" | `{:<50}` | "{}" |"#, self.lint_type, self.message);
            output.push('\n');
            output
        } else {
            format!("{file_with_pos} {} \"{}\"\n", self.lint_type, self.message)
        }
    }
}

#[allow(clippy::result_large_err)]
fn get(path: &str) -> Result<ureq::Response, ureq::Error> {
    const MAX_RETRIES: u8 = 4;
    let mut retries = 0;
    loop {
        match ureq::get(path).call() {
            Ok(res) => return Ok(res),
            Err(e) if retries >= MAX_RETRIES => return Err(e),
            Err(ureq::Error::Transport(e)) => eprintln!("Error: {e}"),
            Err(e) => return Err(e),
        }
        eprintln!("retrying in {retries} seconds...");
        thread::sleep(Duration::from_secs(u64::from(retries)));
        retries += 1;
    }
}

impl CrateSource {
    /// Makes the sources available on the disk for clippy to check.
    /// Clones a git repo and checks out the specified commit or downloads a crate from crates.io or
    /// copies a local folder
    fn download_and_extract(&self) -> Crate {
        match self {
            CrateSource::CratesIo { name, version, options } => {
                let extract_dir = PathBuf::from(LINTCHECK_SOURCES);
                let krate_download_dir = PathBuf::from(LINTCHECK_DOWNLOADS);

                // url to download the crate from crates.io
                let url = format!("https://crates.io/api/v1/crates/{name}/{version}/download");
                println!("Downloading and extracting {name} {version} from {url}");
                create_dirs(&krate_download_dir, &extract_dir);

                let krate_file_path = krate_download_dir.join(format!("{name}-{version}.crate.tar.gz"));
                // don't download/extract if we already have done so
                if !krate_file_path.is_file() {
                    // create a file path to download and write the crate data into
                    let mut krate_dest = std::fs::File::create(&krate_file_path).unwrap();
                    let mut krate_req = get(&url).unwrap().into_reader();
                    // copy the crate into the file
                    std::io::copy(&mut krate_req, &mut krate_dest).unwrap();

                    // unzip the tarball
                    let ungz_tar = flate2::read::GzDecoder::new(std::fs::File::open(&krate_file_path).unwrap());
                    // extract the tar archive
                    let mut archive = tar::Archive::new(ungz_tar);
                    archive.unpack(&extract_dir).expect("Failed to extract!");
                }
                // crate is extracted, return a new Krate object which contains the path to the extracted
                // sources that clippy can check
                Crate {
                    version: version.clone(),
                    name: name.clone(),
                    path: extract_dir.join(format!("{name}-{version}/")),
                    options: options.clone(),
                }
            },
            CrateSource::Git {
                name,
                url,
                commit,
                options,
            } => {
                let repo_path = {
                    let mut repo_path = PathBuf::from(LINTCHECK_SOURCES);
                    // add a -git suffix in case we have the same crate from crates.io and a git repo
                    repo_path.push(format!("{name}-git"));
                    repo_path
                };
                // clone the repo if we have not done so
                if !repo_path.is_dir() {
                    println!("Cloning {url} and checking out {commit}");
                    if !Command::new("git")
                        .arg("clone")
                        .arg(url)
                        .arg(&repo_path)
                        .status()
                        .expect("Failed to clone git repo!")
                        .success()
                    {
                        eprintln!("Failed to clone {url} into {}", repo_path.display());
                    }
                }
                // check out the commit/branch/whatever
                if !Command::new("git")
                    .args(["-c", "advice.detachedHead=false"])
                    .arg("checkout")
                    .arg(commit)
                    .current_dir(&repo_path)
                    .status()
                    .expect("Failed to check out commit")
                    .success()
                {
                    eprintln!("Failed to checkout {commit} of repo at {}", repo_path.display());
                }

                Crate {
                    version: commit.clone(),
                    name: name.clone(),
                    path: repo_path,
                    options: options.clone(),
                }
            },
            CrateSource::Path { name, path, options } => {
                fn is_cache_dir(entry: &DirEntry) -> bool {
                    std::fs::read(entry.path().join("CACHEDIR.TAG"))
                        .map(|x| x.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55"))
                        .unwrap_or(false)
                }

                // copy path into the dest_crate_root but skip directories that contain a CACHEDIR.TAG file.
                // The target/ directory contains a CACHEDIR.TAG file so it is the most commonly skipped directory
                // as a result of this filter.
                let dest_crate_root = PathBuf::from(LINTCHECK_SOURCES).join(name);
                if dest_crate_root.exists() {
                    println!("Deleting existing directory at {dest_crate_root:?}");
                    std::fs::remove_dir_all(&dest_crate_root).unwrap();
                }

                println!("Copying {path:?} to {dest_crate_root:?}");

                for entry in WalkDir::new(path).into_iter().filter_entry(|e| !is_cache_dir(e)) {
                    let entry = entry.unwrap();
                    let entry_path = entry.path();
                    let relative_entry_path = entry_path.strip_prefix(path).unwrap();
                    let dest_path = dest_crate_root.join(relative_entry_path);
                    let metadata = entry_path.symlink_metadata().unwrap();

                    if metadata.is_dir() {
                        std::fs::create_dir(dest_path).unwrap();
                    } else if metadata.is_file() {
                        std::fs::copy(entry_path, dest_path).unwrap();
                    }
                }

                Crate {
                    version: String::from("local"),
                    name: name.clone(),
                    path: dest_crate_root,
                    options: options.clone(),
                }
            },
        }
    }
}

impl Crate {
    /// Run `cargo clippy` on the `Crate` and collect and return all the lint warnings that clippy
    /// issued
    #[allow(clippy::too_many_arguments)]
    fn run_clippy_lints(
        &self,
        cargo_clippy_path: &Path,
        clippy_driver_path: &Path,
        target_dir_index: &AtomicUsize,
        total_crates_to_lint: usize,
        config: &LintcheckConfig,
        lint_filter: &Vec<String>,
        server: &Option<LintcheckServer>,
    ) -> Vec<ClippyWarning> {
        // advance the atomic index by one
        let index = target_dir_index.fetch_add(1, Ordering::SeqCst);
        // "loop" the index within 0..thread_limit
        let thread_index = index % config.max_jobs;
        let perc = (index * 100) / total_crates_to_lint;

        if config.max_jobs == 1 {
            println!(
                "{index}/{total_crates_to_lint} {perc}% Linting {} {}",
                &self.name, &self.version
            );
        } else {
            println!(
                "{index}/{total_crates_to_lint} {perc}% Linting {} {} in target dir {thread_index:?}",
                &self.name, &self.version
            );
        }

        let cargo_clippy_path = std::fs::canonicalize(cargo_clippy_path).unwrap();

        let shared_target_dir = clippy_project_root().join("target/lintcheck/shared_target_dir");

        let mut cargo_clippy_args = if config.fix {
            vec!["--fix", "--"]
        } else {
            vec!["--", "--message-format=json", "--"]
        };

        let mut clippy_args = Vec::<&str>::new();
        if let Some(options) = &self.options {
            for opt in options {
                clippy_args.push(opt);
            }
        } else {
            clippy_args.extend(["-Wclippy::pedantic", "-Wclippy::cargo"]);
        }

        if lint_filter.is_empty() {
            clippy_args.push("--cap-lints=warn");
        } else {
            clippy_args.push("--cap-lints=allow");
            clippy_args.extend(lint_filter.iter().map(std::string::String::as_str));
        }

        if let Some(server) = server {
            let target = shared_target_dir.join("recursive");

            // `cargo clippy` is a wrapper around `cargo check` that mainly sets `RUSTC_WORKSPACE_WRAPPER` to
            // `clippy-driver`. We do the same thing here with a couple changes:
            //
            // `RUSTC_WRAPPER` is used instead of `RUSTC_WORKSPACE_WRAPPER` so that we can lint all crate
            // dependencies rather than only workspace members
            //
            // The wrapper is set to the `lintcheck` so we can force enable linting and ignore certain crates
            // (see `crate::driver`)
            let status = Command::new("cargo")
                .arg("check")
                .arg("--quiet")
                .current_dir(&self.path)
                .env("CLIPPY_ARGS", clippy_args.join("__CLIPPY_HACKERY__"))
                .env("CARGO_TARGET_DIR", target)
                .env("RUSTC_WRAPPER", env::current_exe().unwrap())
                // Pass the absolute path so `crate::driver` can find `clippy-driver`, as it's executed in various
                // different working directories
                .env("CLIPPY_DRIVER", clippy_driver_path)
                .env("LINTCHECK_SERVER", server.local_addr.to_string())
                .status()
                .expect("failed to run cargo");

            assert_eq!(status.code(), Some(0));

            return Vec::new();
        }

        cargo_clippy_args.extend(clippy_args);

        let all_output = Command::new(&cargo_clippy_path)
            // use the looping index to create individual target dirs
            .env("CARGO_TARGET_DIR", shared_target_dir.join(format!("_{thread_index:?}")))
            .args(&cargo_clippy_args)
            .current_dir(&self.path)
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "Encountered error:\n{error:?}\ncargo_clippy_path: {}\ncrate path:{}\n",
                    &cargo_clippy_path.display(),
                    &self.path.display()
                );
            });
        let stdout = String::from_utf8_lossy(&all_output.stdout);
        let stderr = String::from_utf8_lossy(&all_output.stderr);
        let status = &all_output.status;

        if !status.success() {
            eprintln!(
                "\nWARNING: bad exit status after checking {} {} \n",
                self.name, self.version
            );
        }

        if config.fix {
            if let Some(stderr) = stderr
                .lines()
                .find(|line| line.contains("failed to automatically apply fixes suggested by rustc to crate"))
            {
                let subcrate = &stderr[63..];
                println!(
                    "ERROR: failed to apply some suggetion to {} / to (sub)crate {subcrate}",
                    self.name
                );
            }
            // fast path, we don't need the warnings anyway
            return Vec::new();
        }

        // get all clippy warnings and ICEs
        let warnings: Vec<ClippyWarning> = Message::parse_stream(stdout.as_bytes())
            .filter_map(|msg| match msg {
                Ok(Message::CompilerMessage(message)) => ClippyWarning::new(message.message, &self.name, &self.version),
                _ => None,
            })
            .collect();

        warnings
    }
}

/// Builds clippy inside the repo to make sure we have a clippy executable we can use.
fn build_clippy() {
    let status = Command::new("cargo")
        .arg("build")
        .status()
        .expect("Failed to build clippy!");
    if !status.success() {
        eprintln!("Error: Failed to compile Clippy!");
        std::process::exit(1);
    }
}

/// Read a `lintcheck_crates.toml` file
fn read_crates(toml_path: &Path) -> (Vec<CrateSource>, RecursiveOptions) {
    let toml_content: String =
        std::fs::read_to_string(toml_path).unwrap_or_else(|_| panic!("Failed to read {}", toml_path.display()));
    let crate_list: SourceList =
        toml::from_str(&toml_content).unwrap_or_else(|e| panic!("Failed to parse {}: \n{e}", toml_path.display()));
    // parse the hashmap of the toml file into a list of crates
    let tomlcrates: Vec<TomlCrate> = crate_list.crates.into_values().collect();

    // flatten TomlCrates into CrateSources (one TomlCrates may represent several versions of a crate =>
    // multiple Cratesources)
    let mut crate_sources = Vec::new();
    for tk in tomlcrates {
        if let Some(ref path) = tk.path {
            crate_sources.push(CrateSource::Path {
                name: tk.name.clone(),
                path: PathBuf::from(path),
                options: tk.options.clone(),
            });
        } else if let Some(ref versions) = tk.versions {
            // if we have multiple versions, save each one
            for ver in versions.iter() {
                crate_sources.push(CrateSource::CratesIo {
                    name: tk.name.clone(),
                    version: ver.to_string(),
                    options: tk.options.clone(),
                });
            }
        } else if tk.git_url.is_some() && tk.git_hash.is_some() {
            // otherwise, we should have a git source
            crate_sources.push(CrateSource::Git {
                name: tk.name.clone(),
                url: tk.git_url.clone().unwrap(),
                commit: tk.git_hash.clone().unwrap(),
                options: tk.options.clone(),
            });
        } else {
            panic!("Invalid crate source: {tk:?}");
        }

        // if we have a version as well as a git data OR only one git data, something is funky
        if tk.versions.is_some() && (tk.git_url.is_some() || tk.git_hash.is_some())
            || tk.git_hash.is_some() != tk.git_url.is_some()
        {
            eprintln!("tomlkrate: {tk:?}");
            assert_eq!(
                tk.git_hash.is_some(),
                tk.git_url.is_some(),
                "Error: Encountered TomlCrate with only one of git_hash and git_url!"
            );
            assert!(
                tk.path.is_none() || (tk.git_hash.is_none() && tk.versions.is_none()),
                "Error: TomlCrate can only have one of 'git_.*', 'version' or 'path' fields"
            );
            unreachable!("Failed to translate TomlCrate into CrateSource!");
        }
    }
    // sort the crates
    crate_sources.sort();

    (crate_sources, crate_list.recursive)
}

/// Generate a short list of occurring lints-types and their count
fn gather_stats(clippy_warnings: &[ClippyWarning]) -> (String, HashMap<&String, usize>) {
    // count lint type occurrences
    let mut counter: HashMap<&String, usize> = HashMap::new();
    clippy_warnings
        .iter()
        .for_each(|wrn| *counter.entry(&wrn.lint_type).or_insert(0) += 1);

    // collect into a tupled list for sorting
    let mut stats: Vec<(&&String, &usize)> = counter.iter().map(|(lint, count)| (lint, count)).collect();
    // sort by "000{count} {clippy::lintname}"
    // to not have a lint with 200 and 2 warnings take the same spot
    stats.sort_by_key(|(lint, count)| format!("{count:0>4}, {lint}"));

    let mut header = String::from("| lint                                               | count |\n");
    header.push_str("| -------------------------------------------------- | ----- |\n");
    let stats_string = stats
        .iter()
        .map(|(lint, count)| format!("| {lint:<50} |  {count:>4} |\n"))
        .fold(header, |mut table, line| {
            table.push_str(&line);
            table
        });

    (stats_string, counter)
}

#[allow(clippy::too_many_lines)]
fn main() {
    // We're being executed as a `RUSTC_WRAPPER` as part of `--recursive`
    if let Ok(addr) = env::var("LINTCHECK_SERVER") {
        driver::drive(&addr);
    }

    // assert that we launch lintcheck from the repo root (via cargo lintcheck)
    if std::fs::metadata("lintcheck/Cargo.toml").is_err() {
        eprintln!("lintcheck needs to be run from clippy's repo root!\nUse `cargo lintcheck` alternatively.");
        std::process::exit(3);
    }

    let config = LintcheckConfig::new();

    println!("Compiling clippy...");
    build_clippy();
    println!("Done compiling");

    let cargo_clippy_path = fs::canonicalize(format!("target/debug/cargo-clippy{EXE_SUFFIX}")).unwrap();
    let clippy_driver_path = fs::canonicalize(format!("target/debug/clippy-driver{EXE_SUFFIX}")).unwrap();

    // assert that clippy is found
    assert!(
        cargo_clippy_path.is_file(),
        "target/debug/cargo-clippy binary not found! {}",
        cargo_clippy_path.display()
    );

    let clippy_ver = std::process::Command::new(&cargo_clippy_path)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .expect("could not get clippy version!");

    // download and extract the crates, then run clippy on them and collect clippy's warnings
    // flatten into one big list of warnings

    let (crates, recursive_options) = read_crates(&config.sources_toml_path);
    let old_stats = read_stats_from_file(&config.lintcheck_results_path);

    let counter = AtomicUsize::new(1);
    let lint_filter: Vec<String> = config
        .lint_filter
        .iter()
        .map(|filter| {
            let mut filter = filter.clone();
            filter.insert_str(0, "--force-warn=");
            filter
        })
        .collect();

    let crates: Vec<Crate> = crates
        .into_iter()
        .filter(|krate| {
            if let Some(only_one_crate) = &config.only {
                let name = match krate {
                    CrateSource::CratesIo { name, .. }
                    | CrateSource::Git { name, .. }
                    | CrateSource::Path { name, .. } => name,
                };

                name == only_one_crate
            } else {
                true
            }
        })
        .map(|krate| krate.download_and_extract())
        .collect();

    if crates.is_empty() {
        eprintln!(
            "ERROR: could not find crate '{}' in lintcheck/lintcheck_crates.toml",
            config.only.unwrap(),
        );
        std::process::exit(1);
    }

    // run parallel with rayon

    // This helps when we check many small crates with dep-trees that don't have a lot of branches in
    // order to achieve some kind of parallelism

    rayon::ThreadPoolBuilder::new()
        .num_threads(config.max_jobs)
        .build_global()
        .unwrap();

    let server = config.recursive.then(|| {
        let _: io::Result<()> = fs::remove_dir_all("target/lintcheck/shared_target_dir/recursive");

        LintcheckServer::spawn(recursive_options)
    });

    let mut clippy_warnings: Vec<ClippyWarning> = crates
        .par_iter()
        .flat_map(|krate| {
            krate.run_clippy_lints(
                &cargo_clippy_path,
                &clippy_driver_path,
                &counter,
                crates.len(),
                &config,
                &lint_filter,
                &server,
            )
        })
        .collect();

    if let Some(server) = server {
        clippy_warnings.extend(server.warnings());
    }

    // if we are in --fix mode, don't change the log files, terminate here
    if config.fix {
        return;
    }

    // generate some stats
    let (stats_formatted, new_stats) = gather_stats(&clippy_warnings);

    // grab crashes/ICEs, save the crate name and the ice message
    let ices: Vec<(&String, &String)> = clippy_warnings
        .iter()
        .filter(|warning| warning.is_ice)
        .map(|w| (&w.crate_name, &w.message))
        .collect();

    let mut all_msgs: Vec<String> = clippy_warnings
        .iter()
        .map(|warn| warn.to_output(config.markdown))
        .collect();
    all_msgs.sort();
    all_msgs.push("\n\n### Stats:\n\n".into());
    all_msgs.push(stats_formatted);

    // save the text into lintcheck-logs/logs.txt
    let mut text = clippy_ver; // clippy version number on top
    text.push_str("\n### Reports\n\n");
    if config.markdown {
        text.push_str("| file | lint | message |\n");
        text.push_str("| --- | --- | --- |\n");
    }
    write!(text, "{}", all_msgs.join("")).unwrap();
    text.push_str("\n\n### ICEs:\n");
    for (cratename, msg) in &ices {
        let _: fmt::Result = write!(text, "{cratename}: '{msg}'");
    }

    println!("Writing logs to {}", config.lintcheck_results_path.display());
    fs::create_dir_all(config.lintcheck_results_path.parent().unwrap()).unwrap();
    fs::write(&config.lintcheck_results_path, text).unwrap();

    print_stats(old_stats, new_stats, &config.lint_filter);
}

/// read the previous stats from the lintcheck-log file
fn read_stats_from_file(file_path: &Path) -> HashMap<String, usize> {
    let file_content: String = match std::fs::read_to_string(file_path).ok() {
        Some(content) => content,
        None => {
            return HashMap::new();
        },
    };

    let lines: Vec<String> = file_content.lines().map(ToString::to_string).collect();

    lines
        .iter()
        .skip_while(|line| line.as_str() != "### Stats:")
        // Skipping the table header and the `Stats:` label
        .skip(4)
        .take_while(|line| line.starts_with("| "))
        .filter_map(|line| {
            let mut spl = line.split('|');
            // Skip the first `|` symbol
            spl.next();
            if let (Some(lint), Some(count)) = (spl.next(), spl.next()) {
                Some((lint.trim().to_string(), count.trim().parse::<usize>().unwrap()))
            } else {
                None
            }
        })
        .collect::<HashMap<String, usize>>()
}

/// print how lint counts changed between runs
fn print_stats(old_stats: HashMap<String, usize>, new_stats: HashMap<&String, usize>, lint_filter: &Vec<String>) {
    let same_in_both_hashmaps = old_stats
        .iter()
        .filter(|(old_key, old_val)| new_stats.get::<&String>(old_key) == Some(old_val))
        .map(|(k, v)| (k.to_string(), *v))
        .collect::<Vec<(String, usize)>>();

    let mut old_stats_deduped = old_stats;
    let mut new_stats_deduped = new_stats;

    // remove duplicates from both hashmaps
    for (k, v) in &same_in_both_hashmaps {
        assert!(old_stats_deduped.remove(k) == Some(*v));
        assert!(new_stats_deduped.remove(k) == Some(*v));
    }

    println!("\nStats:");

    // list all new counts  (key is in new stats but not in old stats)
    new_stats_deduped
        .iter()
        .filter(|(new_key, _)| old_stats_deduped.get::<str>(new_key).is_none())
        .for_each(|(new_key, new_value)| {
            println!("{new_key} 0 => {new_value}");
        });

    // list all changed counts (key is in both maps but value differs)
    new_stats_deduped
        .iter()
        .filter(|(new_key, _new_val)| old_stats_deduped.get::<str>(new_key).is_some())
        .for_each(|(new_key, new_val)| {
            let old_val = old_stats_deduped.get::<str>(new_key).unwrap();
            println!("{new_key} {old_val} => {new_val}");
        });

    // list all gone counts (key is in old status but not in new stats)
    old_stats_deduped
        .iter()
        .filter(|(old_key, _)| new_stats_deduped.get::<&String>(old_key).is_none())
        .filter(|(old_key, _)| lint_filter.is_empty() || lint_filter.contains(old_key))
        .for_each(|(old_key, old_value)| {
            println!("{old_key} {old_value} => 0");
        });
}

/// Create necessary directories to run the lintcheck tool.
///
/// # Panics
///
/// This function panics if creating one of the dirs fails.
fn create_dirs(krate_download_dir: &Path, extract_dir: &Path) {
    std::fs::create_dir("target/lintcheck/").unwrap_or_else(|err| {
        assert_eq!(
            err.kind(),
            ErrorKind::AlreadyExists,
            "cannot create lintcheck target dir"
        );
    });
    std::fs::create_dir(krate_download_dir).unwrap_or_else(|err| {
        assert_eq!(err.kind(), ErrorKind::AlreadyExists, "cannot create crate download dir");
    });
    std::fs::create_dir(extract_dir).unwrap_or_else(|err| {
        assert_eq!(
            err.kind(),
            ErrorKind::AlreadyExists,
            "cannot create crate extraction dir"
        );
    });
}

/// Returns the path to the Clippy project directory
#[must_use]
fn clippy_project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

#[test]
fn lintcheck_test() {
    let args = [
        "run",
        "--target-dir",
        "lintcheck/target",
        "--manifest-path",
        "./lintcheck/Cargo.toml",
        "--",
        "--crates-toml",
        "lintcheck/test_sources.toml",
    ];
    let status = std::process::Command::new("cargo")
        .args(args)
        .current_dir("..") // repo root
        .status();
    //.output();

    assert!(status.unwrap().success());
}
