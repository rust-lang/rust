//! Gets metadata about a workspace from Cargo

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Describes how this module can fail
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O Error: {0:?}")]
    Io(#[from] std::io::Error),
    #[error("Failed get output from cargo-metadata: {0:?}")]
    GettingMetadata(#[from] cargo_metadata::Error),
    #[error("Failed to run cargo vendor: {0:?}")]
    LaunchingVendor(std::io::Error),
    #[error("Failed to complete cargo vendor")]
    RunningVendor,
    #[error("Bad path {0:?} whilst scraping files")]
    Scraping(PathBuf),
}

/// Uniquely describes a package on crates.io
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Package {
    /// The name of the package
    pub name: String,
    /// The version number
    pub version: String,
}

/// Extra data about a package
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PackageMetadata {
    /// The license it is under
    pub license: String,
    /// The list of authors from the package metadata
    pub authors: Vec<String>,
    /// A list of important files from the package, with their contents.
    ///
    /// This includes *COPYRIGHT*, *NOTICE*, *AUTHOR*, *LICENSE*, and *LICENCE* files, case-insensitive.
    pub notices: BTreeMap<String, String>,
    /// If this is true, this dep is in the Rust Standard Library
    pub is_in_libstd: Option<bool>,
}

/// Use `cargo metadata` and `cargo vendor` to get a list of dependencies and their license data.
///
/// This will involve running `cargo vendor` into `vendor_path` so we can
/// grab the license files.
///
/// Any dependency with a path beginning with `root_path` is ignored, as we
/// assume `reuse` has covered it already.
pub fn get_metadata_and_notices(
    cargo: &Path,
    vendor_path: &Path,
    root_path: &Path,
    manifest_paths: &[&Path],
) -> Result<BTreeMap<Package, PackageMetadata>, Error> {
    let mut output = get_metadata(cargo, root_path, manifest_paths)?;

    // Now do a cargo-vendor and grab everything
    println!("Vendoring deps into {}...", vendor_path.display());
    run_cargo_vendor(cargo, &vendor_path, manifest_paths)?;

    // Now for each dependency we found, go and grab any important looking files
    for (package, metadata) in output.iter_mut() {
        load_important_files(package, metadata, &vendor_path)?;
    }

    Ok(output)
}

/// Use `cargo metadata` to get a list of dependencies and their license data.
///
/// Any dependency with a path beginning with `root_path` is ignored, as we
/// assume `reuse` has covered it already.
pub fn get_metadata(
    cargo: &Path,
    root_path: &Path,
    manifest_paths: &[&Path],
) -> Result<BTreeMap<Package, PackageMetadata>, Error> {
    let mut output = BTreeMap::new();
    // Look at the metadata for each manifest
    for manifest_path in manifest_paths {
        if manifest_path.file_name() != Some(OsStr::new("Cargo.toml")) {
            panic!("cargo_manifest::get requires a path to a Cargo.toml file");
        }
        let metadata = cargo_metadata::MetadataCommand::new()
            .cargo_path(cargo)
            .env("RUSTC_BOOTSTRAP", "1")
            .manifest_path(manifest_path)
            .exec()?;
        for package in metadata.packages {
            let manifest_path = package.manifest_path.as_path();
            if manifest_path.starts_with(root_path) {
                // it's an in-tree dependency and reuse covers it
                continue;
            }
            // otherwise it's an out-of-tree dependency
            let package_id = Package { name: package.name, version: package.version.to_string() };
            output.insert(package_id, PackageMetadata {
                license: package.license.unwrap_or_else(|| String::from("Unspecified")),
                authors: package.authors,
                notices: BTreeMap::new(),
                is_in_libstd: None,
            });
        }
    }

    Ok(output)
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

    let vendor_status = vendor_command.status().map_err(Error::LaunchingVendor)?;

    if !vendor_status.success() {
        return Err(Error::RunningVendor);
    }

    Ok(())
}

/// Add important files off disk into this dependency.
///
/// Maybe one-day Cargo.toml will contain enough information that we don't need
/// to do this manual scraping.
fn load_important_files(
    package: &Package,
    dep: &mut PackageMetadata,
    vendor_root: &Path,
) -> Result<(), Error> {
    let name_version = format!("{}-{}", package.name, package.version);
    println!("Scraping notices for {}...", name_version);
    let dep_vendor_path = vendor_root.join(name_version);
    for entry in std::fs::read_dir(dep_vendor_path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let path = entry.path();
        let Some(filename) = path.file_name() else {
            return Err(Error::Scraping(path));
        };
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
                for inner_entry in std::fs::read_dir(entry.path())? {
                    let inner_entry = inner_entry?;
                    if inner_entry.metadata()?.is_file() {
                        let inner_filename = inner_entry.file_name();
                        let inner_filename_str = inner_filename.to_string_lossy();
                        let qualified_filename =
                            format!("{}/{}", lc_filename_str, inner_filename_str);
                        println!("Scraping {}", qualified_filename);
                        dep.notices.insert(
                            qualified_filename.to_string(),
                            std::fs::read_to_string(inner_entry.path())?,
                        );
                    }
                }
            } else if metadata.is_file() {
                let filename = filename.to_string_lossy();
                println!("Scraping {}", filename);
                dep.notices.insert(filename.to_string(), std::fs::read_to_string(path)?);
            }
        }
    }
    Ok(())
}
