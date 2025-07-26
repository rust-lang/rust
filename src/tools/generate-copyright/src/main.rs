use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Error;
use askama::Template;

mod cargo_metadata;

/// The entry point to the binary.
///
/// You should probably let `bootstrap` execute this program instead of running
/// it directly. It assumes that the current working directory is the root of a
/// Rust git repository checkout, and constructs a bunch of relative paths based
/// on that assumption.
///
/// Run `x.py run generate-copyright`
fn main() -> Result<(), Error> {
    let cargo_home = env_path("CARGO_HOME")?;
    let dest_file = env_path("DEST")?;
    let libstd_dest_file = env_path("DEST_LIBSTD")?;
    let src_dir = env_path("SRC_DIR")?;
    let vendor_dir = env_path("VENDOR_DIR")?;
    let cargo = env_path("CARGO")?;
    let license_metadata = env_path("LICENSE_METADATA")?;

    let cargo_manifests = env_string("CARGO_MANIFESTS")?
        .split(",")
        .map(|manifest| manifest.into())
        .collect::<Vec<PathBuf>>();
    let library_manifests = cargo_manifests
        .iter()
        .filter(|path| {
            if let Ok(stripped) = path.strip_prefix(&src_dir) {
                stripped.starts_with("library")
            } else {
                panic!("manifest {path:?} not relative to source dir {src_dir:?}");
            }
        })
        .cloned()
        .collect::<Vec<_>>();

    // Scan Cargo dependencies
    let mut collected_cargo_metadata = cargo_metadata::get_metadata_and_notices(
        &cargo,
        &cargo_home,
        &vendor_dir,
        &src_dir,
        &cargo_manifests,
    )?;

    let library_collected_cargo_metadata = cargo_metadata::get_metadata_and_notices(
        &cargo,
        &cargo_home,
        &vendor_dir,
        &src_dir,
        &library_manifests,
    )?;

    for (key, value) in collected_cargo_metadata.iter_mut() {
        value.is_in_libstd = Some(library_collected_cargo_metadata.contains_key(key));
    }

    // Load JSON output by reuse
    let collected_tree_metadata: Metadata =
        serde_json::from_slice(&std::fs::read(&license_metadata)?)?;

    // Find libstd sub-set
    let library_collected_tree_metadata = Metadata {
        files: collected_tree_metadata
            .files
            .trim_clone(&src_dir.join("library"), &src_dir)
            .unwrap(),
    };

    // Output main file
    let template = CopyrightTemplate {
        in_tree: collected_tree_metadata.files,
        dependencies: collected_cargo_metadata,
    };
    let output = template.render()?;
    // Git stores text files with \n, but this file may contain \r\n in files
    // copied from dependencies. Normalise them before we write them out, for
    // consistency.
    let output = output.replace("\r\n", "\n");
    std::fs::write(&dest_file, output)?;

    // Output libstd subset file
    let template = LibraryCopyrightTemplate {
        in_tree: library_collected_tree_metadata.files,
        dependencies: library_collected_cargo_metadata,
    };
    let output = template.render()?;
    // Git stores text files with \n, but this file may contain \r\n in files
    // copied from dependencies. Normalise them before we write them out, for
    // consistency.
    let output = output.replace("\r\n", "\n");
    std::fs::write(&libstd_dest_file, output)?;

    Ok(())
}

/// The HTML template for the toolchain copyright file
#[derive(Template)]
#[template(path = "COPYRIGHT.html")]
struct CopyrightTemplate {
    in_tree: Node,
    dependencies: BTreeMap<cargo_metadata::Package, cargo_metadata::PackageMetadata>,
}

/// The HTML template for the library copyright file
#[derive(Template)]
#[template(path = "COPYRIGHT-library.html")]
struct LibraryCopyrightTemplate {
    in_tree: Node,
    dependencies: BTreeMap<cargo_metadata::Package, cargo_metadata::PackageMetadata>,
}

/// Describes a tree of metadata for our filesystem tree
///
/// Must match the JSON emitted by the `CollectLicenseMetadata` bootstrap tool.
#[derive(serde::Deserialize, Clone, Debug, PartialEq, Eq)]
struct Metadata {
    files: Node,
}

/// Describes one node in our metadata tree
#[derive(serde::Deserialize, Template, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "kebab-case", tag = "type")]
#[template(path = "Node.html")]
pub(crate) enum Node {
    Root { children: Vec<Node> },
    Directory { name: String, children: Vec<Node>, license: Option<License> },
    File { name: String, license: License },
    Group { files: Vec<String>, directories: Vec<String>, license: License },
}

impl Node {
    /// Clone, this node, but only if the path to the item is within the match path
    fn trim_clone(&self, match_path: &Path, parent_path: &Path) -> Option<Node> {
        match self {
            Node::Root { children } => {
                let mut filtered_children = Vec::new();
                for node in children {
                    if let Some(child_node) = node.trim_clone(match_path, parent_path) {
                        filtered_children.push(child_node);
                    }
                }
                if filtered_children.is_empty() {
                    None
                } else {
                    Some(Node::Root { children: filtered_children })
                }
            }
            Node::Directory { name, children, license } => {
                let child_name = parent_path.join(name);
                if !(child_name.starts_with(match_path) || match_path.starts_with(&child_name)) {
                    return None;
                }
                let mut filtered_children = Vec::new();
                for node in children {
                    if let Some(child_node) = node.trim_clone(match_path, &child_name) {
                        filtered_children.push(child_node);
                    }
                }
                Some(Node::Directory {
                    name: name.clone(),
                    children: filtered_children,
                    license: license.clone(),
                })
            }
            Node::File { name, license } => {
                let child_name = parent_path.join(name);
                if !(child_name.starts_with(match_path) || match_path.starts_with(&child_name)) {
                    return None;
                }
                Some(Node::File { name: name.clone(), license: license.clone() })
            }
            Node::Group { files, directories, license } => {
                let mut filtered_child_files = Vec::new();
                for child in files {
                    let child_name = parent_path.join(child);
                    if child_name.starts_with(match_path) || match_path.starts_with(&child_name) {
                        filtered_child_files.push(child.clone());
                    }
                }
                let mut filtered_child_dirs = Vec::new();
                for child in directories {
                    let child_name = parent_path.join(child);
                    if child_name.starts_with(match_path) || match_path.starts_with(&child_name) {
                        filtered_child_dirs.push(child.clone());
                    }
                }
                Some(Node::Group {
                    files: filtered_child_files,
                    directories: filtered_child_dirs,
                    license: license.clone(),
                })
            }
        }
    }
}

/// A License has an SPDX license name and a list of copyright holders.
#[derive(serde::Deserialize, Clone, Debug, PartialEq, Eq)]
struct License {
    spdx: String,
    copyright: Vec<String>,
}

/// Grab an environment variable as string, or fail nicely.
fn env_string(var: &str) -> Result<String, Error> {
    match std::env::var(var) {
        Ok(var) => Ok(var),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("environment variable {var} is not utf-8")
        }
        Err(std::env::VarError::NotPresent) => anyhow::bail!("missing environment variable {var}"),
    }
}

/// Grab an environment variable as a PathBuf, or fail nicely.
fn env_path(var: &str) -> Result<PathBuf, Error> {
    if let Some(var) = std::env::var_os(var) {
        Ok(var.into())
    } else {
        anyhow::bail!("missing environment variable {var}")
    }
}
