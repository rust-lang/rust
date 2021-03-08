// Run clippy on a fixed set of crates and collect the warnings.
// This helps observing the impact clippy changs have on a set of real-world code.
//
// When a new lint is introduced, we can search the results for new warnings and check for false
// positives.

#![cfg(feature = "lintcheck")]
#![allow(clippy::filter_map)]

use crate::clippy_project_root;

use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{env, fmt, fs::write, path::PathBuf};

use clap::ArgMatches;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

impl std::fmt::Display for ClippyWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            r#"{}-{}/{}:{}:{} {} "{}""#,
            &self.crate_name, &self.crate_version, &self.file, &self.line, &self.column, &self.linttype, &self.message
        )
    }
}

impl CrateSource {
    /// Makes the sources available on the disk for clippy to check.
    /// Clones a git repo and checks out the specified commit or downloads a crate from crates.io or
    /// copies a local folder
    fn download_and_extract(&self) -> Crate {
        match self {
            CrateSource::CratesIo { name, version, options } => {
                let extract_dir = PathBuf::from("target/lintcheck/crates");
                let krate_download_dir = PathBuf::from("target/lintcheck/downloads");

                // url to download the crate from crates.io
                let url = format!("https://crates.io/api/v1/crates/{}/{}/download", name, version);
                println!("Downloading and extracting {} {} from {}", name, version, url);
                let _ = std::fs::create_dir("target/lintcheck/");
                let _ = std::fs::create_dir(&krate_download_dir);
                let _ = std::fs::create_dir(&extract_dir);

                let krate_file_path = krate_download_dir.join(format!("{}-{}.crate.tar.gz", name, version));
                // don't download/extract if we already have done so
                if !krate_file_path.is_file() {
                    // create a file path to download and write the crate data into
                    let mut krate_dest = std::fs::File::create(&krate_file_path).unwrap();
                    let mut krate_req = ureq::get(&url).call().unwrap().into_reader();
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
                    let mut repo_path = PathBuf::from("target/lintcheck/crates");
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
                use fs_extra::dir;

                // simply copy the entire directory into our target dir
                let copy_dest = PathBuf::from("target/lintcheck/crates/");

                // the source path of the crate we copied,  ${copy_dest}/crate_name
                let crate_root = copy_dest.join(name); // .../crates/local_crate

                if !crate_root.exists() {
                    println!("Copying {} to {}", path.display(), copy_dest.display());

                    dir::copy(path, &copy_dest, &dir::CopyOptions::new()).expect(&format!(
                        "Failed to copy from {}, to  {}",
                        path.display(),
                        crate_root.display()
                    ));
                } else {
                    println!(
                        "Not copying {} to {}, destination already exists",
                        path.display(),
                        crate_root.display()
                    );
                }

                Crate {
                    version: String::from("local"),
                    name: name.clone(),
                    path: crate_root,
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
        cargo_clippy_path: &PathBuf,
        target_dir_index: &AtomicUsize,
        thread_limit: usize,
        total_crates_to_lint: usize,
    ) -> Vec<ClippyWarning> {
        // advance the atomic index by one
        let index = target_dir_index.fetch_add(1, Ordering::SeqCst);
        // "loop" the index within 0..thread_limit
        let target_dir_index = index % thread_limit;
        let perc = ((index * 100) as f32 / total_crates_to_lint as f32) as u8;

        if thread_limit == 1 {
            println!(
                "{}/{} {}% Linting {} {}",
                index, total_crates_to_lint, perc, &self.name, &self.version
            );
        } else {
            println!(
                "{}/{} {}% Linting {} {} in target dir {:?}",
                index, total_crates_to_lint, perc, &self.name, &self.version, target_dir_index
            );
        }

        let cargo_clippy_path = std::fs::canonicalize(cargo_clippy_path).unwrap();

        let shared_target_dir = clippy_project_root().join("target/lintcheck/shared_target_dir");

        let mut args = vec!["--", "--message-format=json", "--", "--cap-lints=warn"];

        if let Some(options) = &self.options {
            for opt in options {
                args.push(opt);
            }
        } else {
            args.extend(&["-Wclippy::pedantic", "-Wclippy::cargo"])
        }

        let all_output = std::process::Command::new(&cargo_clippy_path)
            // use the looping index to create individual target dirs
            .env(
                "CARGO_TARGET_DIR",
                shared_target_dir.join(format!("_{:?}", target_dir_index)),
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

/// get the path to lintchecks crate sources .toml file, check LINTCHECK_TOML first but if it's
/// empty use the default path
fn lintcheck_config_toml(toml_path: Option<&str>) -> PathBuf {
    PathBuf::from(
        env::var("LINTCHECK_TOML").unwrap_or(
            toml_path
                .clone()
                .unwrap_or("clippy_dev/lintcheck_crates.toml")
                .to_string(),
        ),
    )
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
fn read_crates(toml_path: Option<&str>) -> (String, Vec<CrateSource>) {
    let toml_path = lintcheck_config_toml(toml_path);
    // save it so that we can use the name of the sources.toml as name for the logfile later.
    let toml_filename = toml_path.file_stem().unwrap().to_str().unwrap().to_string();
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

    (toml_filename, crate_sources)
}

/// Parse the json output of clippy and return a `ClippyWarning`
fn parse_json_message(json_message: &str, krate: &Crate) -> ClippyWarning {
    let jmsg: Value = serde_json::from_str(&json_message).unwrap_or_else(|e| panic!("Failed to parse json:\n{:?}", e));

    ClippyWarning {
        crate_name: krate.name.to_string(),
        crate_version: krate.version.to_string(),
        file: jmsg["message"]["spans"][0]["file_name"]
            .to_string()
            .trim_matches('"')
            .into(),
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

/// Generate a short list of occuring lints-types and their count
fn gather_stats(clippy_warnings: &[ClippyWarning]) -> String {
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

    stats
        .iter()
        .map(|(lint, count)| format!("{} {}\n", lint, count))
        .collect::<String>()
}

/// check if the latest modification of the logfile is older than the modification date of the
/// clippy binary, if this is true, we should clean the lintchec shared target directory and recheck
fn lintcheck_needs_rerun(toml_path: Option<&str>) -> bool {
    let clippy_modified: std::time::SystemTime = {
        let mut times = ["target/debug/clippy-driver", "target/debug/cargo-clippy"]
            .iter()
            .map(|p| {
                std::fs::metadata(p)
                    .expect("failed to get metadata of file")
                    .modified()
                    .expect("failed to get modification date")
            });
        // the lates modification of either of the binaries
        std::cmp::max(times.next().unwrap(), times.next().unwrap())
    };

    let logs_modified: std::time::SystemTime = std::fs::metadata(lintcheck_config_toml(toml_path))
        .expect("failed to get metadata of file")
        .modified()
        .expect("failed to get modification date");

    // if clippys modification time is bigger (older) than the logs mod time, we need to rerun lintcheck
    clippy_modified > logs_modified
}

/// lintchecks `main()` function
pub fn run(clap_config: &ArgMatches) {
    println!("Compiling clippy...");
    build_clippy();
    println!("Done compiling");

    let clap_toml_path = clap_config.value_of("crates-toml");

    // if the clippy bin is newer than our logs, throw away target dirs to force clippy to
    // refresh the logs
    if lintcheck_needs_rerun(clap_toml_path) {
        let shared_target_dir = "target/lintcheck/shared_target_dir";
        match std::fs::metadata(&shared_target_dir) {
            Ok(metadata) => {
                if metadata.is_dir() {
                    println!("Clippy is newer than lint check logs, clearing lintcheck shared target dir...");
                    std::fs::remove_dir_all(&shared_target_dir)
                        .expect("failed to remove target/lintcheck/shared_target_dir");
                }
            },
            Err(_) => { // dir probably does not exist, don't remove anything
            },
        }
    }

    let cargo_clippy_path: PathBuf = PathBuf::from("target/debug/cargo-clippy")
        .canonicalize()
        .expect("failed to canonicalize path to clippy binary");

    // assert that clippy is found
    assert!(
        cargo_clippy_path.is_file(),
        "target/debug/cargo-clippy binary not found! {}",
        cargo_clippy_path.display()
    );

    let clippy_ver = std::process::Command::new("target/debug/cargo-clippy")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .expect("could not get clippy version!");

    // download and extract the crates, then run clippy on them and collect clippys warnings
    // flatten into one big list of warnings

    let (filename, crates) = read_crates(clap_toml_path);

    let clippy_warnings: Vec<ClippyWarning> = if let Some(only_one_crate) = clap_config.value_of("only") {
        // if we don't have the specified crate in the .toml, throw an error
        if !crates.iter().any(|krate| {
            let name = match krate {
                CrateSource::CratesIo { name, .. } => name,
                CrateSource::Git { name, .. } => name,
                CrateSource::Path { name, .. } => name,
            };
            name == only_one_crate
        }) {
            eprintln!(
                "ERROR: could not find crate '{}' in clippy_dev/lintcheck_crates.toml",
                only_one_crate
            );
            std::process::exit(1);
        }

        // only check a single crate that was passed via cmdline
        crates
            .into_iter()
            .map(|krate| krate.download_and_extract())
            .filter(|krate| krate.name == only_one_crate)
            .map(|krate| krate.run_clippy_lints(&cargo_clippy_path, &AtomicUsize::new(0), 1, 1))
            .flatten()
            .collect()
    } else {
        let counter = std::sync::atomic::AtomicUsize::new(0);

        // Ask rayon for thread count. Assume that half of that is the number of physical cores
        // Use one target dir for each core so that we can run N clippys in parallel.
        // We need to use different target dirs because cargo would lock them for a single build otherwise,
        // killing the parallelism. However this also means that deps will only be reused half/a
        // quarter of the time which might result in a longer wall clock runtime

        // This helps when we check many small crates with dep-trees that don't have a lot of branches in
        // order to achive some kind of parallelism

        // by default, use a single thread
        let num_cpus = match clap_config.value_of("threads") {
            Some(threads) => {
                let threads: usize = threads
                    .parse()
                    .expect(&format!("Failed to parse '{}' to a digit", threads));
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

        let num_crates = crates.len();

        // check all crates (default)
        crates
            .into_par_iter()
            .map(|krate| krate.download_and_extract())
            .map(|krate| krate.run_clippy_lints(&cargo_clippy_path, &counter, num_cpus, num_crates))
            .flatten()
            .collect()
    };

    // generate some stats
    let stats_formatted = gather_stats(&clippy_warnings);

    // grab crashes/ICEs, save the crate name and the ice message
    let ices: Vec<(&String, &String)> = clippy_warnings
        .iter()
        .filter(|warning| warning.is_ice)
        .map(|w| (&w.crate_name, &w.message))
        .collect();

    let mut all_msgs: Vec<String> = clippy_warnings.iter().map(|warning| warning.to_string()).collect();
    all_msgs.sort();
    all_msgs.push("\n\n\n\nStats\n\n".into());
    all_msgs.push(stats_formatted);

    // save the text into lintcheck-logs/logs.txt
    let mut text = clippy_ver; // clippy version number on top
    text.push_str(&format!("\n{}", all_msgs.join("")));
    text.push_str("ICEs:\n");
    ices.iter()
        .for_each(|(cratename, msg)| text.push_str(&format!("{}: '{}'", cratename, msg)));

    let file = format!("lintcheck-logs/{}_logs.txt", filename);
    println!("Writing logs to {}", file);
    write(file, text).unwrap();
}
