//! Gets metadata about a workspace from Cargo

/// Describes how this module can fail
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to run cargo metadata: {0:?}")]
    Launching(#[from] std::io::Error),
    #[error("Failed get output from cargo metadata: {0:?}")]
    GettingMetadata(String),
    #[error("Failed parse JSON output from cargo metadata: {0:?}")]
    ParsingJson(#[from] serde_json::Error),
    #[error("Failed find expected JSON element {0} in output from cargo metadata")]
    MissingJsonElement(&'static str),
    #[error("Failed find expected JSON element {0} in output from cargo metadata for package {1}")]
    MissingJsonElementForPackage(String, String),
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
}

/// Use `cargo` to get a list of dependencies and their license data
///
/// Any dependency with a path beginning with `root_path` is ignored, as we
/// assume `reuse` has covered it already.
pub fn get(
    cargo: &std::path::Path,
    manifest_path: &std::path::Path,
    root_path: &std::path::Path,
) -> Result<Vec<Dependency>, Error> {
    if manifest_path.file_name() != Some(std::ffi::OsStr::new("Cargo.toml")) {
        panic!("cargo_manifest::get requires a path to a Cargo.toml file");
    }
    let metadata_output = std::process::Command::new(cargo)
        .arg("metadata")
        .arg("--format-version=1")
        .arg("--all-features")
        .arg("--manifest-path")
        .arg(manifest_path)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .map_err(|e| Error::Launching(e))?;
    if !metadata_output.status.success() {
        return Err(Error::GettingMetadata(
            String::from_utf8(metadata_output.stderr).expect("UTF-8 output from cargo"),
        ));
    }
    let metadata_json: serde_json::Value = serde_json::from_slice(&metadata_output.stdout)?;
    let packages = metadata_json["packages"]
        .as_array()
        .ok_or_else(|| Error::MissingJsonElement("packages array"))?;
    let mut v = Vec::new();
    for package in packages {
        let package =
            package.as_object().ok_or_else(|| Error::MissingJsonElement("package object"))?;
        // println!("Package: {}", serde_json::to_string_pretty(package).expect("JSON encoding"));
        let manifest_path = package
            .get("manifest_path")
            .and_then(|v| v.as_str())
            .map(std::path::Path::new)
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

        v.push(Dependency {
            name: name.to_owned(),
            version: version.to_owned(),
            license: license.to_owned(),
            authors,
        })
    }

    Ok(v)
}
