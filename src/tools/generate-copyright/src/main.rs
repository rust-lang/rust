use anyhow::Error;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

mod cargo_metadata;

/// The entry point to the binary.
///
/// You should probably let `bootstrap` execute this program instead of running it directly.
///
/// Run `x.py run generate-metadata`
fn main() -> Result<(), Error> {
    let dest = env_path("DEST")?;
    let cargo = env_path("CARGO")?;
    let license_metadata = env_path("LICENSE_METADATA")?;

    let collected_tree_metadata: Metadata =
        serde_json::from_slice(&std::fs::read(&license_metadata)?)?;

    let mut collected_cargo_metadata = BTreeSet::new();

    let root_path = std::path::absolute(".")?;
    for dep in cargo_metadata::get(&cargo, Path::new("./Cargo.toml"), &root_path)? {
        collected_cargo_metadata.insert(dep);
    }
    for dep in cargo_metadata::get(&cargo, Path::new("./src/tools/cargo/Cargo.toml"), &root_path)? {
        collected_cargo_metadata.insert(dep);
    }
    for dep in cargo_metadata::get(&cargo, Path::new("./library/std/Cargo.toml"), &root_path)? {
        collected_cargo_metadata.insert(dep);
    }

    let mut buffer = Vec::new();

    writeln!(buffer, "# In-tree files")?;
    writeln!(buffer)?;
    writeln!(
        buffer,
        "The following licenses cover the in-tree source files that were used in this release:"
    )?;
    writeln!(buffer)?;
    render_tree_recursive(&collected_tree_metadata.files, &mut buffer, 0)?;

    writeln!(buffer)?;

    writeln!(buffer, "# Out-of-tree files")?;
    writeln!(buffer)?;
    writeln!(
        buffer,
        "The following licenses cover the out-of-tree crates that were used in this release:"
    )?;
    writeln!(buffer)?;
    render_deps(collected_cargo_metadata.iter(), &mut buffer)?;

    std::fs::write(&dest, &buffer)?;

    Ok(())
}

/// Recursively draw the tree of files/folders we found on disk and their licences, as
/// markdown, into the given Vec.
fn render_tree_recursive(
    node: &Node,
    buffer: &mut Vec<u8>,
    depth: usize,
) -> Result<(), Error> {
    let prefix = std::iter::repeat("> ").take(depth + 1).collect::<String>();

    match node {
        Node::Root { children } => {
            for child in children {
                render_tree_recursive(child, buffer, depth)?;
            }
        }
        Node::Directory { name, children, license } => {
            render_tree_license(&prefix, std::iter::once(name), license.iter(), buffer)?;
            if !children.is_empty() {
                writeln!(buffer, "{prefix}")?;
                writeln!(buffer, "{prefix}*Exceptions:*")?;
                for child in children {
                    writeln!(buffer, "{prefix}")?;
                    render_tree_recursive(child, buffer, depth + 1)?;
                }
            }
        }
        Node::CondensedDirectory { name, licenses } => {
            render_tree_license(&prefix, std::iter::once(name), licenses.iter(), buffer)?;
        }
        Node::Group { files, directories, license } => {
            render_tree_license(
                &prefix,
                directories.iter().chain(files.iter()),
                std::iter::once(license),
                buffer,
            )?;
        }
        Node::File { name, license } => {
            render_tree_license(&prefix, std::iter::once(name), std::iter::once(license), buffer)?;
        }
    }

    Ok(())
}

/// Draw a series of sibling files/folders, as markdown, into the given Vec.
fn render_tree_license<'a>(
    prefix: &str,
    names: impl Iterator<Item = &'a String>,
    licenses: impl Iterator<Item = &'a License>,
    buffer: &mut Vec<u8>,
) -> Result<(), Error> {
    let mut spdxs = BTreeSet::new();
    let mut copyrights = BTreeSet::new();
    for license in licenses {
        spdxs.insert(&license.spdx);
        for copyright in &license.copyright {
            copyrights.insert(copyright);
        }
    }

    for name in names {
        writeln!(buffer, "{prefix}**`{name}`**  ")?;
    }
    for spdx in spdxs.iter() {
        writeln!(buffer, "{prefix}License: `{spdx}`  ")?;
    }
    for (i, copyright) in copyrights.iter().enumerate() {
        let suffix = if i == copyrights.len() - 1 { "" } else { "  " };
        writeln!(buffer, "{prefix}Copyright: {copyright}{suffix}")?;
    }

    Ok(())
}

/// Render a list of out-of-tree dependencies as markdown into the given Vec.
fn render_deps<'a, 'b>(
    deps: impl Iterator<Item = &'a cargo_metadata::Dependency>,
    buffer: &'b mut Vec<u8>,
) -> Result<(), Error> {
    writeln!(buffer, "| Package | License | URL | Authors |")?;
    writeln!(buffer, "|---------|---------|-----|---------|")?;
    for dep in deps {
        let authors_list = dep.authors.join(", ").replace("<", "\\<").replace(">", "\\>");
        let url = format!("https://crates.io/crates/{}/{}", dep.name, dep.version);
        writeln!(
            buffer,
            "| {name} {version} | {license} | <{url}> | {authors} |",
            name = dep.name,
            version = dep.version,
            license = dep.license,
            url = url,
            authors = authors_list
        )?;
    }
    Ok(())
}

/// Describes a tree of metadata for our filesystem tree
#[derive(serde::Deserialize)]
struct Metadata {
    files: Node,
}

/// Describes one node in our metadata tree
#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub(crate) enum Node {
    Root { children: Vec<Node> },
    Directory { name: String, children: Vec<Node>, license: Option<License> },
    CondensedDirectory { name: String, licenses: Vec<License> },
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
