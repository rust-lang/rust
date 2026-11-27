use std::fmt::Write;
use std::{env, iter};

use anyhow::bail;
use xshell::{Shell, cmd};

pub(crate) fn get_changelog(
    sh: &Shell,
    changelog_n: usize,
    commit: &str,
    prev_tag: &str,
    today: &str,
) -> anyhow::Result<String> {
    let token = match env::var("GITHUB_TOKEN") {
        Ok(token) => token,
        Err(_) => bail!(
            "Please obtain a personal access token from https://github.com/settings/tokens and set the `GITHUB_TOKEN` environment variable."
        ),
    };

    let git_log = cmd!(sh, "git log {prev_tag}..HEAD --reverse").read()?;
    let mut features = String::new();
    let mut fixes = String::new();
    let mut internal = String::new();
    let mut others = String::new();
    for line in git_log.lines() {
        let line = line.trim_start();
        if let Some(pr_num) = parse_pr_number(line) {
            let accept = "Accept: application/vnd.github.v3+json";
            let authorization = format!("Authorization: token {token}");
            let pr_url = "https://api.github.com/repos/rust-lang/rust-analyzer/issues";

            // we don't use an HTTPS client or JSON parser to keep the build times low
            let pr = pr_num.to_string();
            let cmd = &cmd!(sh, "curl --fail -s -H {accept} -H {authorization} {pr_url}/{pr}");
            let pr_json = match cmd.read() {
                Ok(pr_json) => pr_json,
                Err(e) => {
                    // most likely a rust-lang/rust PR
                    eprintln!("Cannot get info for #{pr}: {e}");
                    continue;
                }
            };

            let pr_title = cmd!(sh, "jq .title").stdin(&pr_json).read()?;
            let pr_title = unescape(&pr_title[1..pr_title.len() - 1]);
            let pr_comment = cmd!(sh, "jq .body").stdin(pr_json).read()?;

            let cmd =
                &cmd!(sh, "curl --fail -s -H {accept} -H {authorization} {pr_url}/{pr}/comments");
            let pr_info = match cmd.read() {
                Ok(comments_json) => {
                    let pr_comments = cmd!(sh, "jq .[].body").stdin(comments_json).read()?;

                    iter::once(pr_comment.as_str())
                        .chain(pr_comments.lines())
                        .rev()
                        .find_map(|it| {
                            let it = unescape(&it[1..it.len() - 1]);
                            it.lines().find_map(parse_changelog_line)
                        })
                        .into_iter()
                        .next()
                }
                Err(e) => {
                    eprintln!("Cannot get comments for #{pr}: {e}");
                    None
                }
            };

            let pr_info = pr_info.unwrap_or_else(|| parse_title_line(&pr_title));
            let s = match pr_info.kind {
                PrKind::Feature => &mut features,
                PrKind::Fix => &mut fixes,
                PrKind::Internal => &mut internal,
                PrKind::Other => &mut others,
                PrKind::Skip => continue,
            };
            writeln!(s, "* pr:{pr_num}[] {}", pr_info.message.as_deref().unwrap_or(&pr_title))
                .unwrap();
        }
    }

    let contents = format!(
        "\
= Changelog #{changelog_n}
:sectanchors:
:experimental:
:page-layout: post

Commit: commit:{commit}[] +
Release: release:{today}[] (`TBD`)

== New Features

{features}

== Fixes

{fixes}

== Internal Improvements

{internal}

== Others

{others}
"
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

fn parse_pr_number(s: &str) -> Option<u32> {
    const GITHUB_PREFIX: &str = "Merge pull request #";
    const HOMU_PREFIX: &str = "Auto merge of #";
    if let Some(s) = s.strip_prefix(GITHUB_PREFIX) {
        let s = if let Some(space) = s.find(' ') { &s[..space] } else { s };
        s.parse().ok()
    } else if let Some(s) = s.strip_prefix(HOMU_PREFIX) {
        if let Some(space) = s.find(' ') { s[..space].parse().ok() } else { None }
    } else {
        None
    }
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

    for (prefix, kind) in PREFIXES {
        if lower.starts_with(prefix) {
            let message = match &kind {
                PrKind::Skip => None,
                _ => Some(s[prefix.len()..].to_string()),
            };
            return PrInfo { message, kind };
        }
    }
    PrInfo { kind: PrKind::Other, message: Some(s.to_owned()) }
}
