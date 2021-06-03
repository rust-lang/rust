use std::fmt::Write;
use std::{env, iter};

use anyhow::{bail, Result};
use xshell::cmd;

pub(crate) fn get_changelog(
    changelog_n: usize,
    commit: &str,
    prev_tag: &str,
    today: &str,
) -> Result<String> {
    let git_log = cmd!("git log {prev_tag}..HEAD --merges --reverse").read()?;
    let mut features = String::new();
    let mut fixes = String::new();
    let mut internal = String::new();
    let mut others = String::new();
    for line in git_log.lines() {
        let line = line.trim_start();
        if let Some(p) = line.find(':') {
            let pr = &line[..p];
            if let Ok(pr_num) = pr.parse::<u32>() {
                let accept = "Accept: application/vnd.github.v3+json";
                let token = match env::var("GITHUB_TOKEN") {
                    Ok(token) => token,
                    Err(_) => bail!("Please obtain a personal access token from https://github.com/settings/tokens and set the `GITHUB_TOKEN` environment variable."),
                };
                let authorization = format!("Authorization: token {}", token);
                let pr_url = "https://api.github.com/repos/rust-analyzer/rust-analyzer/issues";

                // we don't use an HTTPS client or JSON parser to keep the build times low
                let pr_json =
                    cmd!("curl -s -H {accept} -H {authorization} {pr_url}/{pr}").read()?;
                let pr_title = cmd!("jq .title").stdin(&pr_json).read()?;
                let pr_title = unescape(&pr_title[1..pr_title.len() - 1]);
                let pr_comment = cmd!("jq .body").stdin(pr_json).read()?;

                let comments_json =
                    cmd!("curl -s -H {accept} -H {authorization} {pr_url}/{pr}/comments").read()?;
                let pr_comments = cmd!("jq .[].body").stdin(comments_json).read()?;

                let l = iter::once(pr_comment.as_str())
                    .chain(pr_comments.lines())
                    .rev()
                    .find_map(|it| {
                        let it = unescape(&it[1..it.len() - 1]);
                        it.lines().find_map(parse_changelog_line)
                    })
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| parse_title_line(&pr_title));
                let s = match l.kind {
                    PrKind::Feature => &mut features,
                    PrKind::Fix => &mut fixes,
                    PrKind::Internal => &mut internal,
                    PrKind::Other => &mut others,
                    PrKind::Skip => continue,
                };
                writeln!(s, "* pr:{}[] {}", pr_num, l.message.as_deref().unwrap_or(&pr_title))
                    .unwrap();
            }
        }
    }

    let contents = format!(
        "\
= Changelog #{}
:sectanchors:
:page-layout: post

Commit: commit:{}[] +
Release: release:{}[]

== Sponsors

**Become a sponsor:** On https://opencollective.com/rust-analyzer/[OpenCollective] or
https://github.com/sponsors/rust-analyzer[GitHub Sponsors].

== New Features

{}

== Fixes

{}

== Internal Improvements

{}

== Others

{}
",
        changelog_n, commit, today, features, fixes, internal, others
    );
    Ok(contents)
}

#[derive(Clone, Copy)]
enum PrKind {
    Feature,
    Fix,
    Internal,
    Other,
    Skip,
}

struct PrInfo {
    message: Option<String>,
    kind: PrKind,
}

fn unescape(s: &str) -> String {
    s.replace(r#"\""#, "").replace(r#"\n"#, "\n").replace(r#"\r"#, "")
}

fn parse_changelog_line(s: &str) -> Option<PrInfo> {
    let parts = s.splitn(3, ' ').collect::<Vec<_>>();
    if parts.len() < 2 || parts[0] != "changelog" {
        return None;
    }
    let message = parts.get(2).map(|it| it.to_string());
    let kind = match parts[1].trim_end_matches(':') {
        "feature" => PrKind::Feature,
        "fix" => PrKind::Fix,
        "internal" => PrKind::Internal,
        "skip" => PrKind::Skip,
        _ => {
            let kind = PrKind::Other;
            let message = format!("{} {}", parts[1], message.unwrap_or_default());
            return Some(PrInfo { kind, message: Some(message) });
        }
    };
    let res = PrInfo { message, kind };
    Some(res)
}

fn parse_title_line(s: &str) -> PrInfo {
    let lower = s.to_ascii_lowercase();
    const PREFIXES: [(&str, PrKind); 5] = [
        ("feat: ", PrKind::Feature),
        ("feature: ", PrKind::Feature),
        ("fix: ", PrKind::Fix),
        ("internal: ", PrKind::Internal),
        ("minor: ", PrKind::Skip),
    ];

    for &(prefix, kind) in &PREFIXES {
        if lower.starts_with(prefix) {
            let message = match &kind {
                PrKind::Skip => None,
                _ => Some(s[prefix.len()..].to_string()),
            };
            return PrInfo { message, kind };
        }
    }
    PrInfo { kind: PrKind::Other, message: Some(s.to_string()) }
}
