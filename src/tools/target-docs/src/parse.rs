//! Suboptimal half-markdown parser that's just good-enough for this.

use eyre::{bail, OptionExt, Result, WrapErr};
use serde::Deserialize;
use std::{collections::HashMap, fs::DirEntry, path::Path};

#[derive(Debug)]
pub struct ParsedTargetInfoFile {
    pub pattern: String,
    pub maintainers: Vec<String>,
    pub sections: Vec<(String, String)>,
    pub footnotes: HashMap<String, Vec<String>>,
}

// IMPORTANT: This is also documented in the README, keep it in sync.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frontmatter {
    #[serde(default)]
    maintainers: Vec<String>,
    #[serde(default)]
    footnotes: HashMap<String, Vec<String>>,
}

// IMPORTANT: This is also documented in the README, keep it in sync.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetFootnotes {
    pub target: String,
    #[serde(default)]
    pub footnotes: Vec<String>,
}

// IMPORTANT: This is also documented in the README, keep it in sync.
#[derive(Debug, PartialEq, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriStateBool {
    True,
    False,
    Unknown,
}

pub fn load_target_infos(directory: &Path) -> Result<Vec<ParsedTargetInfoFile>> {
    let dir = std::fs::read_dir(directory).unwrap();
    let mut infos = Vec::new();

    for entry in dir {
        let entry = entry?;
        infos.push(
            load_single_target_info(&entry)
                .wrap_err_with(|| format!("loading {}", entry.path().display()))?,
        )
    }

    Ok(infos)
}

fn load_single_target_info(entry: &DirEntry) -> Result<ParsedTargetInfoFile> {
    let pattern = entry.file_name();
    let name = pattern
        .to_str()
        .ok_or_eyre("file name is invalid utf8")?
        .strip_suffix(".md")
        .ok_or_eyre("target_info files must end with .md")?;
    let content: String = std::fs::read_to_string(entry.path()).wrap_err("reading content")?;

    parse_file(name, &content)
}

fn parse_file(name: &str, content: &str) -> Result<ParsedTargetInfoFile> {
    let mut frontmatter_splitter = content.split("---\n");

    let frontmatter = frontmatter_splitter.nth(1).ok_or_eyre("missing frontmatter")?;

    let frontmatter_line_count = frontmatter.lines().count() + 2; // 2 from ---

    let frontmatter =
        serde_yaml::from_str::<Frontmatter>(frontmatter).wrap_err("invalid frontmatter")?;

    let body = frontmatter_splitter.next().ok_or_eyre("no body")?;

    let mut sections = Vec::<(String, String)>::new();
    let mut in_codeblock = false;

    for (idx, line) in body.lines().enumerate() {
        let number = frontmatter_line_count + idx + 1; // 1 because "line numbers" are off by 1

        let push_line = |sections: &mut Vec<(String, String)>, line| {
            match sections.last_mut() {
                Some((_, content)) => {
                    content.push_str(line);
                    content.push('\n');
                }
                None if line.trim().is_empty() => {}
                None => {
                    bail!("line {number} with content not allowed before the first heading")
                }
            }
            Ok(())
        };

        if line.starts_with("```") {
            in_codeblock ^= true; // toggle
            push_line(&mut sections, line)?;
        } else if line.starts_with('#') {
            if in_codeblock {
                push_line(&mut sections, line)?;
            } else if let Some(header) = line.strip_prefix("## ") {
                if !crate::SECTIONS.contains(&header) {
                    bail!(
                        "on line {number}, `{header}` is not an allowed section name, must be one of {:?}",
                        super::SECTIONS
                    );
                }
                sections.push((header.to_owned(), String::new()));
            } else {
                bail!("on line {number}, the only allowed headings are `## `: `{line}`");
            }
        } else {
            push_line(&mut sections, line)?;
        }
    }

    sections.iter_mut().for_each(|section| section.1 = section.1.trim().to_owned());

    Ok(ParsedTargetInfoFile {
        pattern: name.to_owned(),
        maintainers: frontmatter.maintainers,
        sections,
        footnotes: frontmatter.footnotes,
    })
}

#[cfg(test)]
mod tests;
