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
use std::{fmt, fs::write, path::PathBuf};

use clap::ArgMatches;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// use this to store the crates when interacting with the crates.toml file
#[derive(Debug, Serialize, Deserialize)]
struct CrateList {
    crates: HashMap<String, TomlCrate>,
}

// crate data we stored in the toml, can have multiple versions per crate
// A single TomlCrate is laster mapped to several CrateSources in that case
#[derive(Debug, Serialize, Deserialize)]
struct TomlCrate {
    name: String,
    versions: Option<Vec<String>>,
    git_url: Option<String>,
    git_hash: Option<String>,
    path: Option<String>,
}

// represents an archive we download from crates.io, or a git repo, or a local repo
#[derive(Debug, Serialize, Deserialize, Eq, Hash, PartialEq)]
enum CrateSource {
    CratesIo { name: String, version: String },
    Git { name: String, url: String, commit: String },
    Path { name: String, path: PathBuf },
}

// represents the extracted sourcecode of a crate
// we actually don't need to special-case git repos here because it does not matter for clippy, yay!
// (clippy only needs a simple path)
#[derive(Debug)]
struct Crate {
    version: String,
    name: String,
    // path to the extracted sources that clippy can check
    path: PathBuf,
}

#[derive(Debug)]
struct ClippyWarning {
    crate_name: String,
    crate_version: String,
    file: String,
    line: String,
    column: String,
    linttype: String,
    message: String,
    ice: bool,
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
    fn download_and_extract(&self) -> Crate {
        match self {
            CrateSource::CratesIo { name, version } => {
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
                }
            },
            CrateSource::Git { name, url, commit } => {
                let repo_path = {
                    let mut repo_path = PathBuf::from("target/lintcheck/crates");
                    // add a -git suffix in case we have the same crate from crates.io and a git repo
                    repo_path.push(format!("{}-git", name));
                    repo_path
                };
                // clone the repo if we have not done so
                if !repo_path.is_dir() {
                    println!("Cloning {} and checking out {}", url, commit);
                    Command::new("git")
                        .arg("clone")
                        .arg(url)
                        .arg(&repo_path)
                        .output()
                        .expect("Failed to clone git repo!");
                }
                // check out the commit/branch/whatever
                Command::new("git")
                    .arg("checkout")
                    .arg(commit)
                    .output()
                    .expect("Failed to check out commit");

                Crate {
                    version: commit.clone(),
                    name: name.clone(),
                    path: repo_path,
                }
            },
            CrateSource::Path { name, path } => {
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
                }
            },
        }
    }
}

impl Crate {
    fn run_clippy_lints(&self, cargo_clippy_path: &PathBuf) -> Vec<ClippyWarning> {
        println!("Linting {} {}...", &self.name, &self.version);
        let cargo_clippy_path = std::fs::canonicalize(cargo_clippy_path).unwrap();

        let shared_target_dir = clippy_project_root().join("target/lintcheck/shared_target_dir/");

        let all_output = std::process::Command::new(&cargo_clippy_path)
            .env("CARGO_TARGET_DIR", shared_target_dir)
            // lint warnings will look like this:
            // src/cargo/ops/cargo_compile.rs:127:35: warning: usage of `FromIterator::from_iter`
            .args(&[
                "--",
                "--message-format=json",
                "--",
                "--cap-lints=warn",
                "-Wclippy::pedantic",
                "-Wclippy::cargo",
            ])
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
            .filter(|line| line.contains("clippy::") || line.contains("internal compiler error: "))
            .map(|json_msg| parse_json_message(json_msg, &self))
            .collect();
        warnings
    }
}

fn build_clippy() {
    Command::new("cargo")
        .arg("build")
        .output()
        .expect("Failed to build clippy!");
}

// get a list of CrateSources we want to check from a "lintcheck_crates.toml" file.
fn read_crates(toml_path: Option<&str>) -> (String, Vec<CrateSource>) {
    let toml_path = PathBuf::from(toml_path.unwrap_or("clippy_dev/lintcheck_crates.toml"));
    // save it so that we can use the name of the sources.toml as name for the logfile later.
    let toml_filename = toml_path.file_stem().unwrap().to_str().unwrap().to_string();
    let toml_content: String =
        std::fs::read_to_string(&toml_path).unwrap_or_else(|_| panic!("Failed to read {}", toml_path.display()));
    let crate_list: CrateList =
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
            });
        }

        // if we have multiple versions, save each one
        if let Some(ref versions) = tk.versions {
            versions.iter().for_each(|ver| {
                crate_sources.push(CrateSource::CratesIo {
                    name: tk.name.clone(),
                    version: ver.to_string(),
                });
            })
        }
        // otherwise, we should have a git source
        if tk.git_url.is_some() && tk.git_hash.is_some() {
            crate_sources.push(CrateSource::Git {
                name: tk.name.clone(),
                url: tk.git_url.clone().unwrap(),
                commit: tk.git_hash.clone().unwrap(),
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
    (toml_filename, crate_sources)
}

// extract interesting data from a json lint message
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
        ice: json_message.contains("internal compiler error: "),
    }
}

// the main fn
pub fn run(clap_config: &ArgMatches) {
    let cargo_clippy_path: PathBuf = PathBuf::from("target/debug/cargo-clippy");

    println!("Compiling clippy...");
    build_clippy();
    println!("Done compiling");

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

    let (filename, crates) = read_crates(clap_config.value_of("crates-toml"));

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
            .map(|krate| krate.run_clippy_lints(&cargo_clippy_path))
            .flatten()
            .collect()
    } else {
        // check all crates (default)
        crates
            .into_iter()
            .map(|krate| krate.download_and_extract())
            .map(|krate| krate.run_clippy_lints(&cargo_clippy_path))
            .flatten()
            .collect()
    };

    // generate some stats:

    // grab crashes/ICEs, save the crate name and the ice message
    let ices: Vec<(&String, &String)> = clippy_warnings
        .iter()
        .filter(|warning| warning.ice)
        .map(|w| (&w.crate_name, &w.message))
        .collect();

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

    let stats_formatted: String = stats
        .iter()
        .map(|(lint, count)| format!("{} {}\n", lint, count))
        .collect::<String>();

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
    write(file, text).unwrap();
}
