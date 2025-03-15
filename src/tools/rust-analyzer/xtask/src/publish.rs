mod notes;

use crate::flags;
use anyhow::bail;
use std::env;
use xshell::{Shell, cmd};

impl flags::PublishReleaseNotes {
    pub(crate) fn run(self, sh: &Shell) -> anyhow::Result<()> {
        let asciidoc = sh.read_file(&self.changelog)?;
        let mut markdown = notes::convert_asciidoc_to_markdown(std::io::Cursor::new(&asciidoc))?;
        if !markdown.starts_with("# Changelog") {
            bail!("changelog Markdown should start with `# Changelog`");
        }
        const NEWLINES: &str = "\n\n";
        let Some(idx) = markdown.find(NEWLINES) else {
            bail!("missing newlines after changelog title");
        };
        markdown.replace_range(0..idx + NEWLINES.len(), "");

        let file_name = check_file_name(self.changelog)?;
        let tag_name = &file_name[0..10];
        let original_changelog_url = create_original_changelog_url(&file_name);
        let additional_paragraph =
            format!("\nSee also the [changelog post]({original_changelog_url}).");
        markdown.push_str(&additional_paragraph);
        if self.dry_run {
            println!("{markdown}");
        } else {
            update_release(sh, tag_name, &markdown)?;
        }
        Ok(())
    }
}

fn check_file_name<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<String> {
    let file_name = path
        .as_ref()
        .file_name()
        .ok_or_else(|| anyhow::format_err!("file name is not specified as `changelog`"))?
        .to_string_lossy();

    let mut chars = file_name.chars();
    if file_name.len() >= 10
        && chars.next().unwrap().is_ascii_digit()
        && chars.next().unwrap().is_ascii_digit()
        && chars.next().unwrap().is_ascii_digit()
        && chars.next().unwrap().is_ascii_digit()
        && chars.next().unwrap() == '-'
        && chars.next().unwrap().is_ascii_digit()
        && chars.next().unwrap().is_ascii_digit()
        && chars.next().unwrap() == '-'
        && chars.next().unwrap().is_ascii_digit()
        && chars.next().unwrap().is_ascii_digit()
    {
        Ok(file_name.to_string())
    } else {
        bail!("unexpected file name format; no date information prefixed")
    }
}

fn create_original_changelog_url(file_name: &str) -> String {
    let year = &file_name[0..4];
    let month = &file_name[5..7];
    let day = &file_name[8..10];
    let mut stem = &file_name[11..];
    if let Some(stripped) = stem.strip_suffix(".adoc") {
        stem = stripped;
    }
    format!("https://rust-analyzer.github.io/thisweek/{year}/{month}/{day}/{stem}.html")
}

fn update_release(sh: &Shell, tag_name: &str, release_notes: &str) -> anyhow::Result<()> {
    let token = match env::var("GITHUB_TOKEN") {
        Ok(token) => token,
        Err(_) => bail!(
            "Please obtain a personal access token from https://github.com/settings/tokens and set the `GITHUB_TOKEN` environment variable."
        ),
    };
    let accept = "Accept: application/vnd.github+json";
    let authorization = format!("Authorization: Bearer {token}");
    let api_version = "X-GitHub-Api-Version: 2022-11-28";
    let release_url = "https://api.github.com/repos/rust-lang/rust-analyzer/releases";

    let release_json = cmd!(
        sh,
        "curl -sf -H {accept} -H {authorization} -H {api_version} {release_url}/tags/{tag_name}"
    )
    .read()?;
    let release_id = cmd!(sh, "jq .id").stdin(release_json).read()?;

    let mut patch = String::new();
    // note: the GitHub API doesn't update the target commit if the tag already exists
    write_json::object(&mut patch)
        .string("tag_name", tag_name)
        .string("target_commitish", "master")
        .string("name", tag_name)
        .string("body", release_notes)
        .bool("draft", false)
        .bool("prerelease", false);
    let _ = cmd!(
        sh,
        "curl -sf -X PATCH -H {accept} -H {authorization} -H {api_version} {release_url}/{release_id} -d {patch}"
    )
    .read()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_changelog_url_creation() {
        let input = "2019-07-24-changelog-0.adoc";
        let actual = create_original_changelog_url(input);
        let expected = "https://rust-analyzer.github.io/thisweek/2019/07/24/changelog-0.html";
        assert_eq!(actual, expected);
    }
}
