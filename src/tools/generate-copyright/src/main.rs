use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Error;
use rinja::Template;

mod cargo_metadata;

#[derive(Template)]
#[template(path = "COPYRIGHT.html")]
struct CopyrightTemplate {
    in_tree: Node,
    dependencies: BTreeMap<cargo_metadata::Package, cargo_metadata::PackageMetadata>,
}

/// The entry point to the binary.
///
/// You should probably let `bootstrap` execute this program instead of running it directly.
///
/// Run `x.py run generate-copyright`
fn main() -> Result<(), Error> {
    let dest_file = env_path("DEST")?;
    let out_dir = env_path("OUT_DIR")?;
    let cargo = env_path("CARGO")?;
    let license_metadata = env_path("LICENSE_METADATA")?;

    let collected_tree_metadata: Metadata =
        serde_json::from_slice(&std::fs::read(&license_metadata)?)?;

    let root_path = std::path::absolute(".")?;
    let workspace_paths = [
        Path::new("./Cargo.toml"),
        Path::new("./src/tools/cargo/Cargo.toml"),
        Path::new("./library/Cargo.toml"),
    ];
    let mut collected_cargo_metadata =
        cargo_metadata::get_metadata_and_notices(&cargo, &out_dir, &root_path, &workspace_paths)?;

    let stdlib_set =
        cargo_metadata::get_metadata(&cargo, &root_path, &[Path::new("./library/std/Cargo.toml")])?;

    for (key, value) in collected_cargo_metadata.iter_mut() {
        value.is_in_libstd = Some(stdlib_set.contains_key(key));
    }

    let template = CopyrightTemplate {
        in_tree: collected_tree_metadata.files,
        dependencies: collected_cargo_metadata,
    };

    let output = template.render()?;

    std::fs::write(&dest_file, output)?;

    Ok(())
}

/// Describes a tree of metadata for our filesystem tree
#[derive(serde::Deserialize)]
struct Metadata {
    files: Node,
}

/// Describes one node in our metadata tree
#[derive(serde::Deserialize, rinja::Template)]
#[serde(rename_all = "kebab-case", tag = "type")]
#[template(path = "Node.html")]
pub(crate) enum Node {
    Root { children: Vec<Node> },
    Directory { name: String, children: Vec<Node>, license: Option<License> },
    File { name: String, license: License },
    Group { files: Vec<String>, directories: Vec<String>, license: License },
}

/// A License has an SPDX license name and a list of copyright holders.
#[derive(serde::Deserialize)]
struct License {
    spdx: String,
    copyright: Vec<String>,
}

/// Grab an environment variable as a PathBuf, or fail nicely.
fn env_path(var: &str) -> Result<PathBuf, Error> {
    if let Some(var) = std::env::var_os(var) {
        Ok(var.into())
    } else {
        anyhow::bail!("missing environment variable {var}")
    }
}
