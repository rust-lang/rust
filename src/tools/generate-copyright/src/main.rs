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
        Path::new("./library/std/Cargo.toml"),
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
#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub(crate) enum Node {
    Root { children: Vec<Node> },
    Directory { name: String, children: Vec<Node>, license: Option<License> },
    File { name: String, license: License },
    Group { files: Vec<String>, directories: Vec<String>, license: License },
}

fn with_box<F>(fmt: &mut std::fmt::Formatter<'_>, inner: F) -> std::fmt::Result
where
    F: FnOnce(&mut std::fmt::Formatter<'_>) -> std::fmt::Result,
{
    writeln!(fmt, r#"<div style="border:1px solid black; padding: 5px;">"#)?;
    inner(fmt)?;
    writeln!(fmt, "</div>")?;
    Ok(())
}

impl std::fmt::Display for Node {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Root { children } => {
                if children.len() > 1 {
                    with_box(fmt, |f| {
                        for child in children {
                            writeln!(f, "{child}")?;
                        }
                        Ok(())
                    })
                } else {
                    for child in children {
                        writeln!(fmt, "{child}")?;
                    }
                    Ok(())
                }
            }
            Node::Directory { name, children, license } => with_box(fmt, |f| {
                render_tree_license(std::iter::once(name), license.as_ref(), f)?;
                if !children.is_empty() {
                    writeln!(f, "<p><b>Exceptions:</b></p>")?;
                    for child in children {
                        writeln!(f, "{child}")?;
                    }
                }
                Ok(())
            }),
            Node::Group { files, directories, license } => with_box(fmt, |f| {
                render_tree_license(directories.iter().chain(files.iter()), Some(license), f)
            }),
            Node::File { name, license } => {
                with_box(fmt, |f| render_tree_license(std::iter::once(name), Some(license), f))
            }
        }
    }
}

/// Draw a series of sibling files/folders, as HTML, into the given formatter.
fn render_tree_license<'a>(
    names: impl Iterator<Item = &'a String>,
    license: Option<&License>,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    writeln!(f, "<p><b>File/Directory:</b> ")?;
    for name in names {
        writeln!(f, "<code>{}</code>", html_escape::encode_text(&name))?;
    }
    writeln!(f, "</p>")?;

    if let Some(license) = license {
        writeln!(f, "<p><b>License:</b> {}</p>", html_escape::encode_text(&license.spdx))?;
        for copyright in license.copyright.iter() {
            writeln!(f, "<p><b>Copyright:</b> {}</p>", html_escape::encode_text(&copyright))?;
        }
    }

    Ok(())
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
