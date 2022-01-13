// Run clippy on a fixed set of crates and collect the warnings.
// This helps observing the impact clippy changes have on a set of real-world code (and not just our
// testsuite).
//
// When a new lint is introduced, we can search the results for new warnings and check for false
// positives.

#![allow(clippy::collapsible_else_if)]

use std::ffi::OsStr;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::HashMap, io::ErrorKind};
use std::{
    env,
    fs::write,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use clap::{App, Arg, ArgMatches};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use walkdir::{DirEntry, WalkDir};

#[cfg(not(windows))]
const CLIPPY_DRIVER_PATH: &str = "target/debug/clippy-driver";
#[cfg(not(windows))]
const CARGO_CLIPPY_PATH: &str = "target/debug/cargo-clippy";

#[cfg(windows)]
const CLIPPY_DRIVER_PATH: &str = "target/debug/clippy-driver.exe";
#[cfg(windows)]
const CARGO_CLIPPY_PATH: &str = "target/debug/cargo-clippy.exe";

const LINTCHECK_DOWNLOADS: &str = "target/lintcheck/downloads";
const LINTCHECK_SOURCES: &str = "target/lintcheck/sources";

/// List of sources to check, loaded from a .toml file
#[derive(Debug, Serialize, Deserialize)]
struct SourceList {
    crates: HashMap<String, TomlCrate>,
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
    crate_version: String,
    file: String,
    line: String,
    column: String,
    linttype: String,
    message: String,
    is_ice: bool,
}

#[allow(unused)]
impl ClippyWarning {
    fn to_output(&self, markdown: bool) -> String {
        let file = format!("{}-{}/{}", &self.crate_name, &self.crate_version, &self.file);
        let file_with_pos = format!("{}:{}:{}", &file, &self.line, &self.column);
        if markdown {
            let lint = format!("`{}`", self.linttype);

            let mut output = String::from("| ");
            output.push_str(&format!(
                "[`{}`](../target/lintcheck/sources/{}#L{})",
                file_with_pos, file, self.line
            ));
            output.push_str(&format!(r#" | {:<50} | "{}" |"#, lint, self.message));
            output.push('\n');
            output
        } else {
            format!(
                "target/lintcheck/sources/{} {} \"{}\"\n",
                file_with_pos, self.linttype, self.message
            )
        }
    }
}

fn get(path: &str) -> Result<ureq::Response, ureq::Error> {
    const MAX_RETRIES: u8 = 4;
    let mut retries = 0;
    loop {
        match ureq::get(path).call() {
            Ok(res) => return Ok(res),
            Err(e) if retries >= MAX_RETRIES => return Err(e),
            Err(ureq::Error::Transport(e)) => eprintln!("Error: {}", e),
            Err(e) => return Err(e),
        }
        eprintln!("retrying in {} seconds...", retries);
        thread::sleep(Duration::from_secs(retries as u64));
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
                let url = format!("https://crates.io/api/v1/crates/{}/{}/download", name, version);
                println!("Downloading and extracting {} {} from {}", name, version, url);
                create_dirs(&krate_download_dir, &extract_dir);

                let krate_file_path = krate_download_dir.join(format!("{}-{}.crate.tar.gz", name, version));
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
                    path: extract_dir.join(format!("{}-{}/", name, version)),
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
                    repo_path.push(format!("{}-git", name));
                    repo_path
                };
                // clone the repo if we have not done so
                if !repo_path.is_dir() {
                    println!("Cloning {} and checking out {}", url, commit);
                    if !Command::new("git")
                        .arg("clone")
                        .arg(url)
                        .arg(&repo_path)
                        .status()
                        .expect("Failed to clone git repo!")
                        .success()
                    {
                        eprintln!("Failed to clone {} into {}", url, repo_path.display())
                    }
                }
                // check out the commit/branch/whatever
                if !Command::new("git")
                    .arg("checkout")
                    .arg(commit)
                    .current_dir(&repo_path)
                    .status()
                    .expect("Failed to check out commit")
                    .success()
                {
                    eprintln!("Failed to checkout {} of repo at {}", commit, repo_path.display())
                }

                Crate {
                    version: commit.clone(),
                    name: name.clone(),
                    path: repo_path,
                    options: options.clone(),
                }
            },
            CrateSource::Path { name, path, options } => {
                // copy path into the dest_crate_root but skip directories that contain a CACHEDIR.TAG file.
                // The target/ directory contains a CACHEDIR.TAG file so it is the most commonly skipped directory
                // as a result of this filter.
                let dest_crate_root = PathBuf::from(LINTCHECK_SOURCES).join(name);
                if dest_crate_root.exists() {
                    println!("Deleting existing directory at {:?}", dest_crate_root);
                    std::fs::remove_dir_all(&dest_crate_root).unwrap();
                }

                println!("Copying {:?} to {:?}", path, dest_crate_root);

                fn is_cache_dir(entry: &DirEntry) -> bool {
                    std::fs::read(entry.path().join("CACHEDIR.TAG"))
                        .map(|x| x.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55"))
                        .unwrap_or(false)
                }

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
    fn run_clippy_lints(
        &self,
        cargo_clippy_path: &Path,
        target_dir_index: &AtomicUsize,
        thread_limit: usize,
        total_crates_to_lint: usize,
        fix: bool,
        lint_filter: &Vec<String>,
    ) -> Vec<ClippyWarning> {
        // advance the atomic index by one
        let index = target_dir_index.fetch_add(1, Ordering::SeqCst);
        // "loop" the index within 0..thread_limit
        let thread_index = index % thread_limit;
        let perc = (index * 100) / total_crates_to_lint;

        if thread_limit == 1 {
            println!(
                "{}/{} {}% Linting {} {}",
                index, total_crates_to_lint, perc, &self.name, &self.version
            );
        } else {
            println!(
                "{}/{} {}% Linting {} {} in target dir {:?}",
                index, total_crates_to_lint, perc, &self.name, &self.version, thread_index
            );
        }

        let cargo_clippy_path = std::fs::canonicalize(cargo_clippy_path).unwrap();

        let shared_target_dir = clippy_project_root().join("target/lintcheck/shared_target_dir");

        let mut args = if fix {
            vec!["--fix", "--allow-no-vcs", "--"]
        } else {
            vec!["--", "--message-format=json", "--"]
        };

        if let Some(options) = &self.options {
            for opt in options {
                args.push(opt);
            }
        } else {
            args.extend(&["-Wclippy::pedantic", "-Wclippy::cargo"])
        }

        if lint_filter.is_empty() {
            args.push("--cap-lints=warn");
        } else {
            args.push("--cap-lints=allow");
            args.extend(lint_filter.iter().map(|filter| filter.as_str()))
        }

        let all_output = std::process::Command::new(&cargo_clippy_path)
            // use the looping index to create individual target dirs
            .env(
                "CARGO_TARGET_DIR",
                shared_target_dir.join(format!("_{:?}", thread_index)),
            )
            // lint warnings will look like this:
            // src/cargo/ops/cargo_compile.rs:127:35: warning: usage of `FromIterator::from_iter`
            .args(&args)
            .current_dir(&self.path)
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "Encountered error:\n{:?}\ncargo_clippy_path: {}\ncrate path:{}\n",
                    error,
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

        if fix {
            if let Some(stderr) = stderr
                .lines()
                .find(|line| line.contains("failed to automatically apply fixes suggested by rustc to crate"))
            {
                let subcrate = &stderr[63..];
                println!(
                    "ERROR: failed to apply some suggetion to {} / to (sub)crate {}",
                    self.name, subcrate
                );
            }
            // fast path, we don't need the warnings anyway
            return Vec::new();
        }

        let output_lines = stdout.lines();
        let warnings: Vec<ClippyWarning> = output_lines
            .into_iter()
            // get all clippy warnings and ICEs
            .filter(|line| filter_clippy_warnings(&line))
            .map(|json_msg| parse_json_message(json_msg, &self))
            .collect();

        warnings
    }
}

#[derive(Debug)]
struct LintcheckConfig {
    /// max number of jobs to spawn (default 1)
    max_jobs: usize,
    /// we read the sources to check from here
    sources_toml_path: PathBuf,
    /// we save the clippy lint results here
    lintcheck_results_path: PathBuf,
    /// whether to just run --fix and not collect all the warnings
    fix: bool,
    /// A list of lint that this lintcheck run shound focus on
    lint_filter: Vec<String>,
    /// Indicate if the output should support markdown syntax
    markdown: bool,
}

impl LintcheckConfig {
    fn from_clap(clap_config: &ArgMatches) -> Self {
        // first, check if we got anything passed via the LINTCHECK_TOML env var,
        // if not, ask clap if we got any value for --crates-toml  <foo>
        // if not, use the default "lintcheck/lintcheck_crates.toml"
        let sources_toml = env::var("LINTCHECK_TOML").unwrap_or_else(|_| {
            clap_config
                .value_of("crates-toml")
                .clone()
                .unwrap_or("lintcheck/lintcheck_crates.toml")
                .to_string()
        });

        let markdown = clap_config.is_present("markdown");
        let sources_toml_path = PathBuf::from(sources_toml);

        // for the path where we save the lint results, get the filename without extension (so for
        // wasd.toml, use "wasd"...)
        let filename: PathBuf = sources_toml_path.file_stem().unwrap().into();
        let lintcheck_results_path = PathBuf::from(format!(
            "lintcheck-logs/{}_logs.{}",
            filename.display(),
            if markdown { "md" } else { "txt" }
        ));

        // look at the --threads arg, if 0 is passed, ask rayon rayon how many threads it would spawn and
        // use half of that for the physical core count
        // by default use a single thread
        let max_jobs = match clap_config.value_of("threads") {
            Some(threads) => {
                let threads: usize = threads
                    .parse()
                    .unwrap_or_else(|_| panic!("Failed to parse '{}' to a digit", threads));
                if threads == 0 {
                    // automatic choice
                    // Rayon seems to return thread count so half that for core count
                    (rayon::current_num_threads() / 2) as usize
                } else {
                    threads
                }
            },
            // no -j passed, use a single thread
            None => 1,
        };
        let fix: bool = clap_config.is_present("fix");
        let lint_filter: Vec<String> = clap_config
            .values_of("filter")
            .map(|iter| {
                iter.map(|lint_name| {
                    let mut filter = lint_name.replace('_', "-");
                    if !filter.starts_with("clippy::") {
                        filter.insert_str(0, "clippy::");
                    }
                    filter
                })
                .collect()
            })
            .unwrap_or_default();

        LintcheckConfig {
            max_jobs,
            sources_toml_path,
            lintcheck_results_path,
            fix,
            lint_filter,
            markdown,
        }
    }
}

/// takes a single json-formatted clippy warnings and returns true (we are interested in that line)
/// or false (we aren't)
fn filter_clippy_warnings(line: &str) -> bool {
    // we want to collect ICEs because clippy might have crashed.
    // these are summarized later
    if line.contains("internal compiler error: ") {
        return true;
    }
    // in general, we want all clippy warnings
    // however due to some kind of bug, sometimes there are absolute paths
    // to libcore files inside the message
    // or we end up with cargo-metadata output (https://github.com/rust-lang/rust-clippy/issues/6508)

    // filter out these message to avoid unnecessary noise in the logs
    if line.contains("clippy::")
        && !(line.contains("could not read cargo metadata")
            || (line.contains(".rustup") && line.contains("toolchains")))
    {
        return true;
    }
    false
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

/// Read a `toml` file and return a list of `CrateSources` that we want to check with clippy
fn read_crates(toml_path: &Path) -> Vec<CrateSource> {
    let toml_content: String =
        std::fs::read_to_string(&toml_path).unwrap_or_else(|_| panic!("Failed to read {}", toml_path.display()));
    let crate_list: SourceList =
        toml::from_str(&toml_content).unwrap_or_else(|e| panic!("Failed to parse {}: \n{}", toml_path.display(), e));
    // parse the hashmap of the toml file into a list of crates
    let tomlcrates: Vec<TomlCrate> = crate_list
        .crates
        .into_iter()
        .map(|(_cratename, tomlcrate)| tomlcrate)
        .collect();

    // flatten TomlCrates into CrateSources (one TomlCrates may represent several versions of a crate =>
    // multiple Cratesources)
    let mut crate_sources = Vec::new();
    tomlcrates.into_iter().for_each(|tk| {
        if let Some(ref path) = tk.path {
            crate_sources.push(CrateSource::Path {
                name: tk.name.clone(),
                path: PathBuf::from(path),
                options: tk.options.clone(),
            });
        }

        // if we have multiple versions, save each one
        if let Some(ref versions) = tk.versions {
            versions.iter().for_each(|ver| {
                crate_sources.push(CrateSource::CratesIo {
                    name: tk.name.clone(),
                    version: ver.to_string(),
                    options: tk.options.clone(),
                });
            })
        }
        // otherwise, we should have a git source
        if tk.git_url.is_some() && tk.git_hash.is_some() {
            crate_sources.push(CrateSource::Git {
                name: tk.name.clone(),
                url: tk.git_url.clone().unwrap(),
                commit: tk.git_hash.clone().unwrap(),
                options: tk.options.clone(),
            });
        }
        // if we have a version as well as a git data OR only one git data, something is funky
        if tk.versions.is_some() && (tk.git_url.is_some() || tk.git_hash.is_some())
            || tk.git_hash.is_some() != tk.git_url.is_some()
        {
            eprintln!("tomlkrate: {:?}", tk);
            if tk.git_hash.is_some() != tk.git_url.is_some() {
                panic!("Error: Encountered TomlCrate with only one of git_hash and git_url!");
            }
            if tk.path.is_some() && (tk.git_hash.is_some() || tk.versions.is_some()) {
                panic!("Error: TomlCrate can only have one of 'git_.*', 'version' or 'path' fields");
            }
            unreachable!("Failed to translate TomlCrate into CrateSource!");
        }
    });
    // sort the crates
    crate_sources.sort();

    crate_sources
}

/// Parse the json output of clippy and return a `ClippyWarning`
fn parse_json_message(json_message: &str, krate: &Crate) -> ClippyWarning {
    let jmsg: Value = serde_json::from_str(&json_message).unwrap_or_else(|e| panic!("Failed to parse json:\n{:?}", e));

    let file: String = jmsg["message"]["spans"][0]["file_name"]
        .to_string()
        .trim_matches('"')
        .into();

    let file = if file.contains(".cargo") {
        // if we deal with macros, a filename may show the origin of a macro which can be inside a dep from
        // the registry.
        // don't show the full path in that case.

        // /home/matthias/.cargo/registry/src/github.com-1ecc6299db9ec823/syn-1.0.63/src/custom_keyword.rs
        let path = PathBuf::from(file);
        let mut piter = path.iter();
        // consume all elements until we find ".cargo", so that "/home/matthias" is skipped
        let _: Option<&OsStr> = piter.find(|x| x == &std::ffi::OsString::from(".cargo"));
        // collect the remaining segments
        let file = piter.collect::<PathBuf>();
        format!("{}", file.display())
    } else {
        file
    };

    ClippyWarning {
        crate_name: krate.name.to_string(),
        crate_version: krate.version.to_string(),
        file,
        line: jmsg["message"]["spans"][0]["line_start"]
            .to_string()
            .trim_matches('"')
            .into(),
        column: jmsg["message"]["spans"][0]["text"][0]["highlight_start"]
            .to_string()
            .trim_matches('"')
            .into(),
        linttype: jmsg["message"]["code"]["code"].to_string().trim_matches('"').into(),
        message: jmsg["message"]["message"].to_string().trim_matches('"').into(),
        is_ice: json_message.contains("internal compiler error: "),
    }
}

/// Generate a short list of occurring lints-types and their count
fn gather_stats(clippy_warnings: &[ClippyWarning]) -> (String, HashMap<&String, usize>) {
    // count lint type occurrences
    let mut counter: HashMap<&String, usize> = HashMap::new();
    clippy_warnings
        .iter()
        .for_each(|wrn| *counter.entry(&wrn.linttype).or_insert(0) += 1);

    // collect into a tupled list for sorting
    let mut stats: Vec<(&&String, &usize)> = counter.iter().map(|(lint, count)| (lint, count)).collect();
    // sort by "000{count} {clippy::lintname}"
    // to not have a lint with 200 and 2 warnings take the same spot
    stats.sort_by_key(|(lint, count)| format!("{:0>4}, {}", count, lint));

    let mut header = String::from("| lint                                               | count |\n");
    header.push_str("| -------------------------------------------------- | ----- |\n");
    let stats_string = stats
        .iter()
        .map(|(lint, count)| format!("| {:<50} |  {:>4} |\n", lint, count))
        .fold(header, |mut table, line| {
            table.push_str(&line);
            table
        });

    (stats_string, counter)
}

/// check if the latest modification of the logfile is older than the modification date of the
/// clippy binary, if this is true, we should clean the lintchec shared target directory and recheck
fn lintcheck_needs_rerun(lintcheck_logs_path: &Path) -> bool {
    if !lintcheck_logs_path.exists() {
        return true;
    }

    let clippy_modified: std::time::SystemTime = {
        let mut times = [CLIPPY_DRIVER_PATH, CARGO_CLIPPY_PATH].iter().map(|p| {
            std::fs::metadata(p)
                .expect("failed to get metadata of file")
                .modified()
                .expect("failed to get modification date")
        });
        // the oldest modification of either of the binaries
        std::cmp::max(times.next().unwrap(), times.next().unwrap())
    };

    let logs_modified: std::time::SystemTime = std::fs::metadata(lintcheck_logs_path)
        .expect("failed to get metadata of file")
        .modified()
        .expect("failed to get modification date");

    // time is represented in seconds since X
    // logs_modified 2 and clippy_modified 5 means clippy binary is older and we need to recheck
    logs_modified < clippy_modified
}

/// lintchecks `main()` function
///
/// # Panics
///
/// This function panics if the clippy binaries don't exist
/// or if lintcheck is executed from the wrong directory (aka none-repo-root)
pub fn main() {
    // assert that we launch lintcheck from the repo root (via cargo lintcheck)
    if std::fs::metadata("lintcheck/Cargo.toml").is_err() {
        eprintln!("lintcheck needs to be run from clippys repo root!\nUse `cargo lintcheck` alternatively.");
        std::process::exit(3);
    }

    let clap_config = &get_clap_config();

    let config = LintcheckConfig::from_clap(clap_config);

    println!("Compiling clippy...");
    build_clippy();
    println!("Done compiling");

    // if the clippy bin is newer than our logs, throw away target dirs to force clippy to
    // refresh the logs
    if lintcheck_needs_rerun(&config.lintcheck_results_path) {
        let shared_target_dir = "target/lintcheck/shared_target_dir";
        // if we get an Err here, the shared target dir probably does simply not exist
        if let Ok(metadata) = std::fs::metadata(&shared_target_dir) {
            if metadata.is_dir() {
                println!("Clippy is newer than lint check logs, clearing lintcheck shared target dir...");
                std::fs::remove_dir_all(&shared_target_dir)
                    .expect("failed to remove target/lintcheck/shared_target_dir");
            }
        }
    }

    let cargo_clippy_path: PathBuf = PathBuf::from(CARGO_CLIPPY_PATH)
        .canonicalize()
        .expect("failed to canonicalize path to clippy binary");

    // assert that clippy is found
    assert!(
        cargo_clippy_path.is_file(),
        "target/debug/cargo-clippy binary not found! {}",
        cargo_clippy_path.display()
    );

    let clippy_ver = std::process::Command::new(CARGO_CLIPPY_PATH)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .expect("could not get clippy version!");

    // download and extract the crates, then run clippy on them and collect clippys warnings
    // flatten into one big list of warnings

    let crates = read_crates(&config.sources_toml_path);
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

    let clippy_warnings: Vec<ClippyWarning> = if let Some(only_one_crate) = clap_config.value_of("only") {
        // if we don't have the specified crate in the .toml, throw an error
        if !crates.iter().any(|krate| {
            let name = match krate {
                CrateSource::CratesIo { name, .. } | CrateSource::Git { name, .. } | CrateSource::Path { name, .. } => {
                    name
                },
            };
            name == only_one_crate
        }) {
            eprintln!(
                "ERROR: could not find crate '{}' in lintcheck/lintcheck_crates.toml",
                only_one_crate
            );
            std::process::exit(1);
        }

        // only check a single crate that was passed via cmdline
        crates
            .into_iter()
            .map(|krate| krate.download_and_extract())
            .filter(|krate| krate.name == only_one_crate)
            .flat_map(|krate| {
                krate.run_clippy_lints(&cargo_clippy_path, &AtomicUsize::new(0), 1, 1, config.fix, &lint_filter)
            })
            .collect()
    } else {
        if config.max_jobs > 1 {
            // run parallel with rayon

            // Ask rayon for thread count. Assume that half of that is the number of physical cores
            // Use one target dir for each core so that we can run N clippys in parallel.
            // We need to use different target dirs because cargo would lock them for a single build otherwise,
            // killing the parallelism. However this also means that deps will only be reused half/a
            // quarter of the time which might result in a longer wall clock runtime

            // This helps when we check many small crates with dep-trees that don't have a lot of branches in
            // order to achieve some kind of parallelism

            // by default, use a single thread
            let num_cpus = config.max_jobs;
            let num_crates = crates.len();

            // check all crates (default)
            crates
                .into_par_iter()
                .map(|krate| krate.download_and_extract())
                .flat_map(|krate| {
                    krate.run_clippy_lints(
                        &cargo_clippy_path,
                        &counter,
                        num_cpus,
                        num_crates,
                        config.fix,
                        &lint_filter,
                    )
                })
                .collect()
        } else {
            // run sequential
            let num_crates = crates.len();
            crates
                .into_iter()
                .map(|krate| krate.download_and_extract())
                .flat_map(|krate| {
                    krate.run_clippy_lints(&cargo_clippy_path, &counter, 1, num_crates, config.fix, &lint_filter)
                })
                .collect()
        }
    };

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
    text.push_str(&format!("{}", all_msgs.join("")));
    text.push_str("\n\n### ICEs:\n");
    ices.iter()
        .for_each(|(cratename, msg)| text.push_str(&format!("{}: '{}'", cratename, msg)));

    println!("Writing logs to {}", config.lintcheck_results_path.display());
    std::fs::create_dir_all(config.lintcheck_results_path.parent().unwrap()).unwrap();
    write(&config.lintcheck_results_path, text).unwrap();

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
        .filter(|(old_key, old_val)| new_stats.get::<&String>(&old_key) == Some(old_val))
        .map(|(k, v)| (k.to_string(), *v))
        .collect::<Vec<(String, usize)>>();

    let mut old_stats_deduped = old_stats;
    let mut new_stats_deduped = new_stats;

    // remove duplicates from both hashmaps
    same_in_both_hashmaps.iter().for_each(|(k, v)| {
        assert!(old_stats_deduped.remove(k) == Some(*v));
        assert!(new_stats_deduped.remove(k) == Some(*v));
    });

    println!("\nStats:");

    // list all new counts  (key is in new stats but not in old stats)
    new_stats_deduped
        .iter()
        .filter(|(new_key, _)| old_stats_deduped.get::<str>(&new_key).is_none())
        .for_each(|(new_key, new_value)| {
            println!("{} 0 => {}", new_key, new_value);
        });

    // list all changed counts (key is in both maps but value differs)
    new_stats_deduped
        .iter()
        .filter(|(new_key, _new_val)| old_stats_deduped.get::<str>(&new_key).is_some())
        .for_each(|(new_key, new_val)| {
            let old_val = old_stats_deduped.get::<str>(&new_key).unwrap();
            println!("{} {} => {}", new_key, old_val, new_val);
        });

    // list all gone counts (key is in old status but not in new stats)
    old_stats_deduped
        .iter()
        .filter(|(old_key, _)| new_stats_deduped.get::<&String>(&old_key).is_none())
        .filter(|(old_key, _)| lint_filter.is_empty() || lint_filter.contains(old_key))
        .for_each(|(old_key, old_value)| {
            println!("{} {} => 0", old_key, old_value);
        });
}

/// Create necessary directories to run the lintcheck tool.
///
/// # Panics
///
/// This function panics if creating one of the dirs fails.
fn create_dirs(krate_download_dir: &Path, extract_dir: &Path) {
    std::fs::create_dir("target/lintcheck/").unwrap_or_else(|err| {
        if err.kind() != ErrorKind::AlreadyExists {
            panic!("cannot create lintcheck target dir");
        }
    });
    std::fs::create_dir(&krate_download_dir).unwrap_or_else(|err| {
        if err.kind() != ErrorKind::AlreadyExists {
            panic!("cannot create crate download dir");
        }
    });
    std::fs::create_dir(&extract_dir).unwrap_or_else(|err| {
        if err.kind() != ErrorKind::AlreadyExists {
            panic!("cannot create crate extraction dir");
        }
    });
}

fn get_clap_config<'a>() -> ArgMatches<'a> {
    App::new("lintcheck")
        .about("run clippy on a set of crates and check output")
        .arg(
            Arg::with_name("only")
                .takes_value(true)
                .value_name("CRATE")
                .long("only")
                .help("only process a single crate of the list"),
        )
        .arg(
            Arg::with_name("crates-toml")
                .takes_value(true)
                .value_name("CRATES-SOURCES-TOML-PATH")
                .long("crates-toml")
                .help("set the path for a crates.toml where lintcheck should read the sources from"),
        )
        .arg(
            Arg::with_name("threads")
                .takes_value(true)
                .value_name("N")
                .short("j")
                .long("jobs")
                .help("number of threads to use, 0 automatic choice"),
        )
        .arg(
            Arg::with_name("fix")
                .long("--fix")
                .help("runs cargo clippy --fix and checks if all suggestions apply"),
        )
        .arg(
            Arg::with_name("filter")
                .long("--filter")
                .takes_value(true)
                .multiple(true)
                .value_name("clippy_lint_name")
                .help("apply a filter to only collect specified lints, this also overrides `allow` attributes"),
        )
        .arg(
            Arg::with_name("markdown")
                .long("--markdown")
                .help("change the reports table to use markdown links"),
        )
        .get_matches()
}

/// Returns the path to the Clippy project directory
///
/// # Panics
///
/// Panics if the current directory could not be retrieved, there was an error reading any of the
/// Cargo.toml files or ancestor directory is the clippy root directory
#[must_use]
pub fn clippy_project_root() -> PathBuf {
    let current_dir = std::env::current_dir().unwrap();
    for path in current_dir.ancestors() {
        let result = std::fs::read_to_string(path.join("Cargo.toml"));
        if let Err(err) = &result {
            if err.kind() == std::io::ErrorKind::NotFound {
                continue;
            }
        }

        let content = result.unwrap();
        if content.contains("[package]\nname = \"clippy\"") {
            return path.to_path_buf();
        }
    }
    panic!("error: Can't determine root of project. Please run inside a Clippy working dir.");
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
        .args(&args)
        .current_dir("..") // repo root
        .status();
    //.output();

    assert!(status.unwrap().success());
}
