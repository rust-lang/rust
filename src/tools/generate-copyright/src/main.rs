use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Error;

mod cargo_metadata;

static TOP_BOILERPLATE: &str = r##"
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Copyright notices for The Rust Toolchain</title>
</head>
<body>

<h1>Copyright notices for The Rust Toolchain</h1>

<p>This file describes the copyright and licensing information for the source
code within The Rust Project git tree, and the third-party dependencies used
when building the Rust toolchain (including the Rust Standard Library).</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#in-tree-files">In-tree files</a></li>
    <li><a href="#out-of-tree-dependencies">Out-of-tree dependencies</a></li>
</ul>
"##;

static BOTTOM_BOILERPLATE: &str = r#"
</body>
</html>
"#;

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
    let collected_cargo_metadata =
        cargo_metadata::get_metadata_and_notices(&cargo, &out_dir, &root_path, &workspace_paths)?;

    let stdlib_set =
        cargo_metadata::get_metadata(&cargo, &root_path, &[Path::new("./library/std/Cargo.toml")])?;

    let mut buffer = Vec::new();

    writeln!(buffer, "{}", TOP_BOILERPLATE)?;

    writeln!(
        buffer,
        r#"<h2 id="in-tree-files">In-tree files</h2><p>The following licenses cover the in-tree source files that were used in this release:</p>"#
    )?;
    render_tree_recursive(&collected_tree_metadata.files, &mut buffer)?;

    writeln!(
        buffer,
        r#"<h2 id="out-of-tree-dependencies">Out-of-tree dependencies</h2><p>The following licenses cover the out-of-tree crates that were used in this release:</p>"#
    )?;
    render_deps(&collected_cargo_metadata, &stdlib_set, &mut buffer)?;

    writeln!(buffer, "{}", BOTTOM_BOILERPLATE)?;

    std::fs::write(&dest_file, &buffer)?;

    Ok(())
}

/// Recursively draw the tree of files/folders we found on disk and their licenses, as
/// markdown, into the given Vec.
fn render_tree_recursive(node: &Node, buffer: &mut Vec<u8>) -> Result<(), Error> {
    writeln!(buffer, r#"<div style="border:1px solid black; padding: 5px;">"#)?;
    match node {
        Node::Root { children } => {
            for child in children {
                render_tree_recursive(child, buffer)?;
            }
        }
        Node::Directory { name, children, license } => {
            render_tree_license(std::iter::once(name), license.as_ref(), buffer)?;
            if !children.is_empty() {
                writeln!(buffer, "<p><b>Exceptions:</b></p>")?;
                for child in children {
                    render_tree_recursive(child, buffer)?;
                }
            }
        }
        Node::Group { files, directories, license } => {
            render_tree_license(directories.iter().chain(files.iter()), Some(license), buffer)?;
        }
        Node::File { name, license } => {
            render_tree_license(std::iter::once(name), Some(license), buffer)?;
        }
    }
    writeln!(buffer, "</div>")?;

    Ok(())
}

/// Draw a series of sibling files/folders, as markdown, into the given Vec.
fn render_tree_license<'a>(
    names: impl Iterator<Item = &'a String>,
    license: Option<&License>,
    buffer: &mut Vec<u8>,
) -> Result<(), Error> {
    writeln!(buffer, "<p><b>File/Directory:</b> ")?;
    for name in names {
        writeln!(buffer, "<code>{name}</code>")?;
    }
    writeln!(buffer, "</p>")?;

    if let Some(license) = license {
        writeln!(buffer, "<p><b>License:</b> {}</p>", license.spdx)?;
        for copyright in license.copyright.iter() {
            writeln!(buffer, "<p><b>Copyright:</b> {copyright}</p>")?;
        }
    }

    Ok(())
}

/// Render a list of out-of-tree dependencies as markdown into the given Vec.
fn render_deps(
    all_deps: &BTreeMap<cargo_metadata::Package, cargo_metadata::PackageMetadata>,
    stdlib_set: &BTreeMap<cargo_metadata::Package, cargo_metadata::PackageMetadata>,
    buffer: &mut Vec<u8>,
) -> Result<(), Error> {
    for (package, metadata) in all_deps {
        let authors_list = if metadata.authors.is_empty() {
            "None Specified".to_owned()
        } else {
            metadata.authors.join(", ")
        };
        let url = format!("https://crates.io/crates/{}/{}", package.name, package.version);
        writeln!(buffer)?;
        writeln!(
            buffer,
            r#"<h3>📦 {name}-{version}</h3>"#,
            name = package.name,
            version = package.version,
        )?;
        writeln!(buffer, r#"<p><b>URL:</b> <a href="{url}">{url}</a></p>"#,)?;
        writeln!(
            buffer,
            "<p><b>In libstd:</b> {}</p>",
            if stdlib_set.contains_key(package) { "Yes" } else { "No" }
        )?;
        writeln!(buffer, "<p><b>Authors:</b> {}</p>", escape_html(&authors_list))?;
        writeln!(buffer, "<p><b>License:</b> {}</p>", escape_html(&metadata.license))?;
        writeln!(buffer, "<p><b>Notices:</b> ")?;
        if metadata.notices.is_empty() {
            writeln!(buffer, "None")?;
        } else {
            for (name, contents) in &metadata.notices {
                writeln!(
                    buffer,
                    "<details><summary><code>{}</code></summary>",
                    name.to_string_lossy()
                )?;
                writeln!(buffer, "<pre>\n{}\n</pre>", contents)?;
                writeln!(buffer, "</details>")?;
            }
        }
        writeln!(buffer, "</p>")?;
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

/// Escapes any invalid HTML characters
fn escape_html(input: &str) -> String {
    static MAPPING: [(char, &str); 3] = [('&', "&amp;"), ('<', "&lt;"), ('>', "&gt;")];
    let mut output = input.to_owned();
    for (ch, s) in &MAPPING {
        output = output.replace(*ch, s);
    }
    output
}
