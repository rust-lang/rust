use std::collections::{HashMap, HashSet};
use std::fs::{self};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::Deserialize;
use walkdir::{DirEntry, WalkDir};

use crate::{Crate, lintcheck_sources, target_dir};

const DEFAULT_DOCS_LINK: &str = "https://docs.rs/{krate}/{version}/src/{krate_}/{file}.html#{line}";
const DEFAULT_GITHUB_LINK: &str = "{url}/blob/{hash}/src/{file}#L{line}";
const DEFAULT_PATH_LINK: &str = "{path}/src/{file}:{line}";

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
    /// Magic values:
    /// * `{krate}` will be replaced by `self.name`
    /// * `{krate_}` will be replaced by `self.name` with all `-` replaced by `_`
    /// * `{version}` will be replaced by `self.version`
    /// * `{url}` will be replaced with `self.git_url`
    /// * `{hash}` will be replaced with `self.git_hash`
    /// * `{path}` will be replaced with `self.path`
    /// * `{file}` will be replaced by the path after `src/`
    /// * `{line}` will be replaced by the line
    ///
    /// If unset, this will be filled by [`read_crates`] since it depends on
    /// the source.
    online_link: Option<String>,
}

impl TomlCrate {
    fn file_link(&self, default: &str) -> String {
        let mut link = self.online_link.clone().unwrap_or_else(|| default.to_string());
        link = link.replace("{krate}", &self.name);
        link = link.replace("{krate_}", &self.name.replace('-', "_"));

        if let Some(version) = &self.version {
            link = link.replace("{version}", version);
        }
        if let Some(url) = &self.git_url {
            link = link.replace("{url}", url);
        }
        if let Some(hash) = &self.git_hash {
            link = link.replace("{hash}", hash);
        }
        if let Some(path) = &self.path {
            link = link.replace("{path}", path);
        }
        link
    }
}

/// Represents an archive we download from crates.io, or a git repo, or a local repo/folder
/// Once processed (downloaded/extracted/cloned/copied...), this will be translated into a `Crate`
#[derive(Debug, Deserialize, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct CrateWithSource {
    pub name: String,
    pub source: CrateSource,
    pub file_link: String,
    pub options: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub enum CrateSource {
    CratesIo { version: String },
    Git { url: String, commit: String },
    Path { path: PathBuf },
}

/// Read a `lintcheck_crates.toml` file
pub fn read_crates(toml_path: &Path) -> (Vec<CrateWithSource>, RecursiveOptions) {
    let toml_content: String =
        fs::read_to_string(toml_path).unwrap_or_else(|_| panic!("Failed to read {}", toml_path.display()));
    let crate_list: SourceList =
        toml::from_str(&toml_content).unwrap_or_else(|e| panic!("Failed to parse {}: \n{e}", toml_path.display()));
    // parse the hashmap of the toml file into a list of crates
    let toml_crates: Vec<TomlCrate> = crate_list.crates.into_values().collect();

    // flatten TomlCrates into CrateSources (one TomlCrates may represent several versions of a crate =>
    // multiple CrateSources)
    let mut crate_sources = Vec::new();
    for tk in toml_crates {
        if let Some(ref path) = tk.path {
            crate_sources.push(CrateWithSource {
                name: tk.name.clone(),
                source: CrateSource::Path {
                    path: PathBuf::from(path),
                },
                file_link: tk.file_link(DEFAULT_PATH_LINK),
                options: tk.options.clone(),
            });
        } else if let Some(ref version) = tk.version {
            crate_sources.push(CrateWithSource {
                name: tk.name.clone(),
                source: CrateSource::CratesIo {
                    version: version.clone(),
                },
                file_link: tk.file_link(DEFAULT_DOCS_LINK),
                options: tk.options.clone(),
            });
        } else if tk.git_url.is_some() && tk.git_hash.is_some() {
            // otherwise, we should have a git source
            crate_sources.push(CrateWithSource {
                name: tk.name.clone(),
                source: CrateSource::Git {
                    url: tk.git_url.clone().unwrap(),
                    commit: tk.git_hash.clone().unwrap(),
                },
                file_link: tk.file_link(DEFAULT_GITHUB_LINK),
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

impl CrateWithSource {
    pub fn download_and_prepare(&self) -> Crate {
        let krate = self.download_and_extract();

        // Downloaded crates might contain a `rust-toolchain` file. This file
        // seems to be accessed when `build.rs` files are present. This access
        // results in build errors since lintcheck and clippy will most certainly
        // use a different toolchain.
        // Lintcheck simply removes these files and assumes that our toolchain
        // is more up to date.
        let _ = fs::remove_file(krate.path.join("rust-toolchain"));
        let _ = fs::remove_file(krate.path.join("rust-toolchain.toml"));

        krate
    }
    /// Makes the sources available on the disk for clippy to check.
    /// Clones a git repo and checks out the specified commit or downloads a crate from crates.io or
    /// copies a local folder
    #[expect(clippy::too_many_lines)]
    fn download_and_extract(&self) -> Crate {
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
        let name = &self.name;
        let options = &self.options;
        let file_link = &self.file_link;
        match &self.source {
            CrateSource::CratesIo { version } => {
                let extract_dir = PathBuf::from(lintcheck_sources());
                // Keep constant downloads path to avoid repeating work and
                // filling up disk space unnecessarily.
                let krate_download_dir = PathBuf::from("target/lintcheck/downloads/");

                // url to download the crate from crates.io
                let url = format!("https://crates.io/api/v1/crates/{name}/{version}/download");
                println!("Downloading and extracting {name} {version} from {url}");
                create_dirs(&krate_download_dir, &extract_dir);

                let krate_file_path = krate_download_dir.join(format!("{name}-{version}.crate.tar.gz"));
                // don't download/extract if we already have done so
                if !krate_file_path.is_file() || !extract_dir.join(format!("{name}-{version}")).exists() {
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
                    base_url: file_link.clone(),
                }
            },
            CrateSource::Git { url, commit } => {
                let repo_path = {
                    let mut repo_path = PathBuf::from(lintcheck_sources());
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
                    base_url: file_link.clone(),
                }
            },
            CrateSource::Path { path } => {
                fn is_cache_dir(entry: &DirEntry) -> bool {
                    fs::read(entry.path().join("CACHEDIR.TAG"))
                        .map(|x| x.starts_with(b"Signature: 8a477f597d28d172789f06886806bc55"))
                        .unwrap_or(false)
                }

                // copy path into the dest_crate_root but skip directories that contain a CACHEDIR.TAG file.
                // The target/ directory contains a CACHEDIR.TAG file so it is the most commonly skipped directory
                // as a result of this filter.
                let dest_crate_root = PathBuf::from(lintcheck_sources()).join(name);
                if dest_crate_root.exists() {
                    println!("Deleting existing directory at `{}`", dest_crate_root.display());
                    fs::remove_dir_all(&dest_crate_root).unwrap();
                }

                println!("Copying `{}` to `{}`", path.display(), dest_crate_root.display());

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
                    base_url: file_link.clone(),
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
    fs::create_dir(format!("{}/lintcheck/", target_dir())).unwrap_or_else(|err| {
        assert_eq!(
            err.kind(),
            ErrorKind::AlreadyExists,
            "cannot create lintcheck target dir"
        );
    });
    fs::create_dir_all(krate_download_dir).unwrap_or_else(|err| {
        // We are allowed to reuse download dirs
        assert_ne!(err.kind(), ErrorKind::AlreadyExists);
    });
    fs::create_dir(extract_dir).unwrap_or_else(|err| {
        assert_eq!(
            err.kind(),
            ErrorKind::AlreadyExists,
            "cannot create crate extraction dir"
        );
    });
}
