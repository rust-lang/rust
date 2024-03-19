//! Suboptimal half-markdown parser that's just good-enough for this.

use eyre::{bail, OptionExt, Result, WrapErr};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::DirEntry,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub(crate) struct ParsedTargetInfoFile {
    pub(crate) pattern: String,
    pub(crate) pattern_glob: glob::Pattern,
    pub(crate) maintainers: Vec<String>,
    pub(crate) sections: Vec<(String, String)>,
    pub(crate) footnotes: HashMap<String, Vec<String>>,
    /// Only used for error messages.
    pub(crate) full_path: PathBuf,
}

// IMPORTANT: This is also documented in the README, keep it in sync.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frontmatter {
    pattern: String,
    #[serde(default)]
    maintainers: Vec<String>,
    #[serde(default)]
    footnotes: HashMap<String, Vec<String>>,
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

    parse_file(entry.path(), name, &content)
}

fn parse_file(full_path: PathBuf, file_name: &str, content: &str) -> Result<ParsedTargetInfoFile> {
    let mut frontmatter_splitter = content.split("---\n");

    let frontmatter = frontmatter_splitter.nth(1).ok_or_eyre("missing frontmatter")?;

    let frontmatter_line_count = frontmatter.lines().count() + 2; // 2 from ---

    let frontmatter =
        serde_yaml::from_str::<Frontmatter>(frontmatter).wrap_err("invalid frontmatter")?;

    let expected_file_name = frontmatter.pattern.replace('*', "_");
    if expected_file_name != file_name {
        bail!(
            "`target pattern does not match file name. file name must be pattern with `*` replaced with `_`.\n\
            Expected file name `{expected_file_name}.md`"
        );
    }

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

    let pattern_glob =
        glob::Pattern::new(&frontmatter.pattern).wrap_err("pattern is not a valid glob pattern")?;

    Ok(ParsedTargetInfoFile {
        pattern: frontmatter.pattern,
        pattern_glob,
        maintainers: frontmatter.maintainers,
        sections,
        footnotes: frontmatter.footnotes,
        full_path,
    })
}

#[cfg(test)]
mod tests;
