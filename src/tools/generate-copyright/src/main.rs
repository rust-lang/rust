use anyhow::Error;
use std::io::Write;
use std::path::PathBuf;

fn main() -> Result<(), Error> {
    let dest = env_path("DEST")?;
    let license_metadata = env_path("LICENSE_METADATA")?;

    let metadata: Metadata = serde_json::from_slice(&std::fs::read(&license_metadata)?)?;

    let mut buffer = Vec::new();
    render_recursive(&metadata.files, &mut buffer, 0)?;

    std::fs::write(&dest, &buffer)?;

    Ok(())
}

fn render_recursive(node: &Node, buffer: &mut Vec<u8>, depth: usize) -> Result<(), Error> {
    let prefix = std::iter::repeat("> ").take(depth + 1).collect::<String>();

    match node {
        Node::Root { childs } => {
            for child in childs {
                render_recursive(child, buffer, depth)?;
            }
        }
        Node::Directory { name, childs, license } => {
            render_license(&prefix, std::iter::once(name), license, buffer)?;
            if !childs.is_empty() {
                writeln!(buffer, "{prefix}")?;
                writeln!(buffer, "{prefix}*Exceptions:*")?;
                for child in childs {
                    writeln!(buffer, "{prefix}")?;
                    render_recursive(child, buffer, depth + 1)?;
                }
            }
        }
        Node::Group { files, directories, license } => {
            render_license(&prefix, directories.iter().chain(files.iter()), license, buffer)?;
        }
        Node::File { name, license } => {
            render_license(&prefix, std::iter::once(name), license, buffer)?;
        }
    }

    Ok(())
}

fn render_license<'a>(
    prefix: &str,
    names: impl Iterator<Item = &'a String>,
    license: &License,
    buffer: &mut Vec<u8>,
) -> Result<(), Error> {
    for name in names {
        writeln!(buffer, "{prefix}**`{name}`**  ")?;
    }
    writeln!(buffer, "{prefix}License: `{}`  ", license.spdx)?;
    for (i, copyright) in license.copyright.iter().enumerate() {
        let suffix = if i == license.copyright.len() - 1 { "" } else { "  " };
        writeln!(buffer, "{prefix}Copyright: {copyright}{suffix}")?;
    }

    Ok(())
}

#[derive(serde::Deserialize)]
struct Metadata {
    files: Node,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub(crate) enum Node {
    Root { childs: Vec<Node> },
    Directory { name: String, childs: Vec<Node>, license: License },
    File { name: String, license: License },
    Group { files: Vec<String>, directories: Vec<String>, license: License },
}

#[derive(serde::Deserialize)]
struct License {
    spdx: String,
    copyright: Vec<String>,
}

fn env_path(var: &str) -> Result<PathBuf, Error> {
    if let Some(var) = std::env::var_os(var) {
        Ok(var.into())
    } else {
        anyhow::bail!("missing environment variable {var}")
    }
}
