//! Gets metadata about a workspace from Cargo

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::path::Path;

/// Describes how this module can fail
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to run cargo metadata: {0:?}")]
    LaunchingMetadata(#[from] std::io::Error),
    #[error("Failed get output from cargo metadata: {0:?}")]
    GettingMetadata(String),
    #[error("Failed parse JSON output from cargo metadata: {0:?}")]
    ParsingJson(#[from] serde_json::Error),
    #[error("Failed find expected JSON element {0} in output from cargo metadata")]
    MissingJsonElement(&'static str),
    #[error("Failed find expected JSON element {0} in output from cargo metadata for package {1}")]
    MissingJsonElementForPackage(String, String),
    #[error("Failed to run cargo vendor: {0:?}")]
    LaunchingVendor(std::io::Error),
    #[error("Failed to complete cargo vendor")]
    RunningVendor,
}

/// Describes one of our dependencies
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dependency {
    /// The name of the package
    pub name: String,
    /// The version number
    pub version: String,
    /// The license it is under
    pub license: String,
    /// The list of authors from the package metadata
    pub authors: Vec<String>,
    /// A list of important files from the package, with their contents.
    ///
    /// This includes *COPYRIGHT*, *NOTICE*, *AUTHOR*, *LICENSE*, and *LICENCE* files, case-insensitive.
    pub notices: BTreeMap<OsString, String>,
}

/// Use `cargo` to get a list of dependencies and their license data.
///
/// This will involve running `cargo vendor` into `${BUILD}/vendor` so we can
/// grab the license files.
///
/// Any dependency with a path beginning with `root_path` is ignored, as we
/// assume `reuse` has covered it already.
pub fn get(
    cargo: &Path,
    dest: &Path,
    root_path: &Path,
    manifest_paths: &[&Path],
) -> Result<BTreeSet<Dependency>, Error> {
    let mut temp_set = BTreeSet::new();
    // Look at the metadata for each manifest
    for manifest_path in manifest_paths {
        if manifest_path.file_name() != Some(OsStr::new("Cargo.toml")) {
            panic!("cargo_manifest::get requires a path to a Cargo.toml file");
        }
        let metadata_json = get_metadata_json(cargo, manifest_path)?;
        let packages = metadata_json["packages"]
            .as_array()
            .ok_or_else(|| Error::MissingJsonElement("packages array"))?;
        for package in packages {
            let package =
                package.as_object().ok_or_else(|| Error::MissingJsonElement("package object"))?;
            let manifest_path = package
                .get("manifest_path")
                .and_then(|v| v.as_str())
                .map(Path::new)
                .ok_or_else(|| Error::MissingJsonElement("package.manifest_path"))?;
            if manifest_path.starts_with(&root_path) {
                // it's an in-tree dependency and reuse covers it
                continue;
            }
            // otherwise it's an out-of-tree dependency
            let get_string = |field_name: &str, package_name: &str| {
                package.get(field_name).and_then(|v| v.as_str()).ok_or_else(|| {
                    Error::MissingJsonElementForPackage(
                        format!("package.{field_name}"),
                        package_name.to_owned(),
                    )
                })
            };
            let name = get_string("name", "unknown")?;
            let license = get_string("license", name)?;
            let version = get_string("version", name)?;
            let authors_list = package
                .get("authors")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::MissingJsonElement("package.authors"))?;
            let authors: Vec<String> =
                authors_list.iter().filter_map(|v| v.as_str()).map(|s| s.to_owned()).collect();
            temp_set.insert(Dependency {
                name: name.to_owned(),
                version: version.to_owned(),
                license: license.to_owned(),
                authors,
                notices: BTreeMap::new(),
            });
        }
    }

    // Now do a cargo-vendor and grab everything
    let vendor_path = dest.join("vendor");
    println!("Vendoring deps into {}...", vendor_path.display());
    run_cargo_vendor(cargo, &vendor_path, manifest_paths)?;

    // Now for each dependency we found, go and grab any important looking files
    let mut output = BTreeSet::new();
    for mut dep in temp_set {
        load_important_files(&mut dep, &vendor_path)?;
        output.insert(dep);
    }

    Ok(output)
}

/// Get cargo-metdata for a package, as JSON
fn get_metadata_json(cargo: &Path, manifest_path: &Path) -> Result<serde_json::Value, Error> {
    let metadata_output = std::process::Command::new(cargo)
        .arg("metadata")
        .arg("--format-version=1")
        .arg("--all-features")
        .arg("--manifest-path")
        .arg(manifest_path)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .map_err(|e| Error::LaunchingMetadata(e))?;
    if !metadata_output.status.success() {
        return Err(Error::GettingMetadata(
            String::from_utf8(metadata_output.stderr).expect("UTF-8 output from cargo"),
        ));
    }
    let json = serde_json::from_slice(&metadata_output.stdout)?;
    Ok(json)
}

/// Run cargo-vendor, fetching into the given dir
fn run_cargo_vendor(cargo: &Path, dest: &Path, manifest_paths: &[&Path]) -> Result<(), Error> {
    let mut vendor_command = std::process::Command::new(cargo);
    vendor_command.env("RUSTC_BOOTSTRAP", "1");
    vendor_command.arg("vendor");
    vendor_command.arg("--quiet");
    vendor_command.arg("--versioned-dirs");
    for manifest_path in manifest_paths {
        vendor_command.arg("-s");
        vendor_command.arg(manifest_path);
    }
    vendor_command.arg(dest);

    let vendor_status = vendor_command.status().map_err(|e| Error::LaunchingVendor(e))?;

    if !vendor_status.success() {
        return Err(Error::RunningVendor);
    }

    Ok(())
}

/// Add important files off disk into this dependency.
///
/// Maybe one-day Cargo.toml will contain enough information that we don't need
/// to do this manual scraping.
fn load_important_files(dep: &mut Dependency, vendor_root: &Path) -> Result<(), Error> {
    let name_version = format!("{}-{}", dep.name, dep.version);
    println!("Scraping notices for {}...", name_version);
    let dep_vendor_path = vendor_root.join(name_version);
    for entry in std::fs::read_dir(dep_vendor_path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let path = entry.path();
        if let Some(filename) = path.file_name() {
            let lc_filename = filename.to_ascii_lowercase();
            let lc_filename_str = lc_filename.to_string_lossy();
            let mut keep = false;
            for m in ["copyright", "licence", "license", "author", "notice"] {
                if lc_filename_str.contains(m) {
                    keep = true;
                    break;
                }
            }
            if keep {
                if metadata.is_dir() {
                    // scoop up whole directory
                } else if metadata.is_file() {
                    println!("Scraping {}", filename.to_string_lossy());
                    dep.notices.insert(filename.to_owned(), std::fs::read_to_string(path)?);
                }
            }
        }
    }
    Ok(())
}
