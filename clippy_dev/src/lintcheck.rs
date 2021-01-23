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
    crates: HashMap<String, Vec<String>>,
}

// crate data we stored in the toml, can have multiple versions per crate
// A single TomlCrate is laster mapped to several CrateSources in that case
struct TomlCrate {
    name: String,
    versions: Vec<String>,
}

// represents an archive we download from crates.io
#[derive(Debug, Serialize, Deserialize, Eq, Hash, PartialEq)]
struct CrateSource {
    name: String,
    version: String,
}

// represents the extracted sourcecode of a crate
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
        let extract_dir = PathBuf::from("target/lintcheck/crates");
        let krate_download_dir = PathBuf::from("target/lintcheck/downloads");

        // url to download the crate from crates.io
        let url = format!(
            "https://crates.io/api/v1/crates/{}/{}/download",
            self.name, self.version
        );
        println!("Downloading and extracting {} {} from {}", self.name, self.version, url);
        let _ = std::fs::create_dir("target/lintcheck/");
        let _ = std::fs::create_dir(&krate_download_dir);
        let _ = std::fs::create_dir(&extract_dir);

        let krate_file_path = krate_download_dir.join(format!("{}-{}.crate.tar.gz", &self.name, &self.version));
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
            version: self.version.clone(),
            name: self.name.clone(),
            path: extract_dir.join(format!("{}-{}/", self.name, self.version)),
        }
    }
}

impl Crate {
    fn run_clippy_lints(&self, cargo_clippy_path: &PathBuf) -> Vec<ClippyWarning> {
        println!("Linting {} {}...", &self.name, &self.version);
        let cargo_clippy_path = std::fs::canonicalize(cargo_clippy_path).unwrap();

        let shared_target_dir = clippy_project_root().join("target/lintcheck/shared_target_dir/");

        let all_output = std::process::Command::new(cargo_clippy_path)
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
            .unwrap();
        let stdout = String::from_utf8_lossy(&all_output.stdout);
        let output_lines = stdout.lines();
        //dbg!(&output_lines);
        let warnings: Vec<ClippyWarning> = output_lines
            .into_iter()
            // get all clippy warnings
            .filter(|line| line.contains("clippy::"))
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
fn read_crates() -> Vec<CrateSource> {
    let toml_path = PathBuf::from("clippy_dev/lintcheck_crates.toml");
    let toml_content: String =
        std::fs::read_to_string(&toml_path).unwrap_or_else(|_| panic!("Failed to read {}", toml_path.display()));
    let crate_list: CrateList =
        toml::from_str(&toml_content).unwrap_or_else(|e| panic!("Failed to parse {}: \n{}", toml_path.display(), e));
    // parse the hashmap of the toml file into a list of crates
    let tomlcrates: Vec<TomlCrate> = crate_list
        .crates
        .into_iter()
        .map(|(name, versions)| TomlCrate { name, versions })
        .collect();

    // flatten TomlCrates into CrateSources (one TomlCrates may represent several versions of a crate =>
    // multiple Cratesources)
    let mut crate_sources = Vec::new();
    tomlcrates.into_iter().for_each(|tk| {
        tk.versions.iter().for_each(|ver| {
            crate_sources.push(CrateSource {
                name: tk.name.clone(),
                version: ver.to_string(),
            });
        })
    });
    crate_sources
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

    let crates = read_crates();

    let clippy_warnings: Vec<ClippyWarning> = if let Some(only_one_crate) = clap_config.value_of("only") {
        // if we don't have the specified crated in the .toml, throw an error
        if !crates.iter().any(|krate| krate.name == only_one_crate) {
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
    write("lintcheck-logs/logs.txt", text).unwrap();
}
