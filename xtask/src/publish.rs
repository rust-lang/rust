mod notes;

use crate::flags;
use anyhow::{anyhow, bail, Result};
use std::env;
use xshell::{cmd, Shell};

impl flags::PublishReleaseNotes {
    pub(crate) fn run(self, sh: &Shell) -> Result<()> {
        let asciidoc = sh.read_file(&self.changelog)?;
        let markdown = notes::convert_asciidoc_to_markdown(std::io::Cursor::new(&asciidoc))?;
        let tag_name = extract_tag_name(&self.changelog)?;
        if self.dry_run {
            println!("{}", markdown);
        } else {
            update_release(sh, &tag_name, &markdown)?;
        }
        Ok(())
    }
}

fn extract_tag_name<P: AsRef<std::path::Path>>(path: P) -> Result<String> {
    let file_name = path
        .as_ref()
        .file_name()
        .ok_or_else(|| anyhow!("file name is not specified as `changelog`"))?
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
        Ok(file_name[0..10].to_owned())
    } else {
        bail!("extraction of date from the file name failed")
    }
}

fn update_release(sh: &Shell, tag_name: &str, release_notes: &str) -> Result<()> {
    let token = match env::var("GITHUB_TOKEN") {
        Ok(token) => token,
        Err(_) => bail!("Please obtain a personal access token from https://github.com/settings/tokens and set the `GITHUB_TOKEN` environment variable."),
    };
    let accept = "Accept: application/vnd.github+json";
    let authorization = format!("Authorization: Bearer {}", token);
    let api_version = "X-GitHub-Api-Version: 2022-11-28";
    let release_url = "https://api.github.com/repos/rust-lang/rust-analyzer/releases";

    let release_json = cmd!(
        sh,
        "curl -s -H {accept} -H {authorization} -H {api_version} {release_url}/tags/{tag_name}"
    )
    .read()?;
    let release_id = cmd!(sh, "jq .id").stdin(release_json).read()?;

    let mut patch = String::new();
    write_json::object(&mut patch)
        .string("tag_name", &tag_name)
        .string("target_commitish", "master")
        .string("name", &tag_name)
        .string("body", &release_notes)
        .bool("draft", false)
        .bool("prerelease", false);
    let _ = cmd!(
        sh,
        "curl -s -X PATCH -H {accept} -H {authorization} -H {api_version} {release_url}/{release_id} -d {patch}"
    )
    .read()?;

    Ok(())
}
