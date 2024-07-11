use std::collections::{HashMap, HashSet};
use std::fs::{self};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::Deserialize;
use walkdir::{DirEntry, WalkDir};

use crate::{Crate, LINTCHECK_DOWNLOADS, LINTCHECK_SOURCES};

/// List of sources to check, loaded from a .toml file
#[derive(Debug, Deserialize)]
pub struct SourceList {
    crates: HashMap<String, TomlCrate>,
    #[serde(default)]
    recursive: RecursiveOptions,
}

#[derive(Debug, Deserialize, Default)]
pub struct RecursiveOptions {
    pub ignore: HashSet<String>,
}

/// A crate source stored inside the .toml
/// will be translated into on one of the `CrateSource` variants
#[derive(Debug, Deserialize)]
struct TomlCrate {
    name: String,
    version: Option<String>,
    git_url: Option<String>,
    git_hash: Option<String>,
    path: Option<String>,
    options: Option<Vec<String>>,
}

/// Represents an archive we download from crates.io, or a git repo, or a local repo/folder
/// Once processed (downloaded/extracted/cloned/copied...), this will be translated into a `Crate`
#[derive(Debug, Deserialize, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub enum CrateSource {
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

/// Read a `lintcheck_crates.toml` file
pub fn read_crates(toml_path: &Path) -> (Vec<CrateSource>, RecursiveOptions) {
    let toml_content: String =
        fs::read_to_string(toml_path).unwrap_or_else(|_| panic!("Failed to read {}", toml_path.display()));
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
        } else if let Some(ref version) = tk.version {
            crate_sources.push(CrateSource::CratesIo {
                name: tk.name.clone(),
                version: version.to_string(),
                options: tk.options.clone(),
            });
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
        if tk.version.is_some() && (tk.git_url.is_some() || tk.git_hash.is_some())
            || tk.git_hash.is_some() != tk.git_url.is_some()
        {
            eprintln!("tomlkrate: {tk:?}");
            assert_eq!(
                tk.git_hash.is_some(),
                tk.git_url.is_some(),
                "Error: Encountered TomlCrate with only one of git_hash and git_url!"
            );
            assert!(
                tk.path.is_none() || (tk.git_hash.is_none() && tk.version.is_none()),
                "Error: TomlCrate can only have one of 'git_.*', 'version' or 'path' fields"
            );
            unreachable!("Failed to translate TomlCrate into CrateSource!");
        }
    }
    // sort the crates
    crate_sources.sort();

    (crate_sources, crate_list.recursive)
}

impl CrateSource {
    /// Makes the sources available on the disk for clippy to check.
    /// Clones a git repo and checks out the specified commit or downloads a crate from crates.io or
    /// copies a local folder
    #[expect(clippy::too_many_lines)]
    pub fn download_and_extract(&self) -> Crate {
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
                std::thread::sleep(Duration::from_secs(u64::from(retries)));
                retries += 1;
            }
        }
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
                    let mut krate_dest = fs::File::create(&krate_file_path).unwrap();
                    let mut krate_req = get(&url).unwrap().into_reader();
                    // copy the crate into the file
                    io::copy(&mut krate_req, &mut krate_dest).unwrap();

                    // unzip the tarball
                    let ungz_tar = flate2::read::GzDecoder::new(fs::File::open(&krate_file_path).unwrap());
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
                    fs::read(entry.path().join("CACHEDIR.TAG"))
                        .map(|x| x.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55"))
                        .unwrap_or(false)
                }

                // copy path into the dest_crate_root but skip directories that contain a CACHEDIR.TAG file.
                // The target/ directory contains a CACHEDIR.TAG file so it is the most commonly skipped directory
                // as a result of this filter.
                let dest_crate_root = PathBuf::from(LINTCHECK_SOURCES).join(name);
                if dest_crate_root.exists() {
                    println!("Deleting existing directory at {dest_crate_root:?}");
                    fs::remove_dir_all(&dest_crate_root).unwrap();
                }

                println!("Copying {path:?} to {dest_crate_root:?}");

                for entry in WalkDir::new(path).into_iter().filter_entry(|e| !is_cache_dir(e)) {
                    let entry = entry.unwrap();
                    let entry_path = entry.path();
                    let relative_entry_path = entry_path.strip_prefix(path).unwrap();
                    let dest_path = dest_crate_root.join(relative_entry_path);
                    let metadata = entry_path.symlink_metadata().unwrap();

                    if metadata.is_dir() {
                        fs::create_dir(dest_path).unwrap();
                    } else if metadata.is_file() {
                        fs::copy(entry_path, dest_path).unwrap();
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

/// Create necessary directories to run the lintcheck tool.
///
/// # Panics
///
/// This function panics if creating one of the dirs fails.
fn create_dirs(krate_download_dir: &Path, extract_dir: &Path) {
    fs::create_dir("target/lintcheck/").unwrap_or_else(|err| {
        assert_eq!(
            err.kind(),
            ErrorKind::AlreadyExists,
            "cannot create lintcheck target dir"
        );
    });
    fs::create_dir(krate_download_dir).unwrap_or_else(|err| {
        assert_eq!(err.kind(), ErrorKind::AlreadyExists, "cannot create crate download dir");
    });
    fs::create_dir(extract_dir).unwrap_or_else(|err| {
        assert_eq!(
            err.kind(),
            ErrorKind::AlreadyExists,
            "cannot create crate extraction dir"
        );
    });
}
