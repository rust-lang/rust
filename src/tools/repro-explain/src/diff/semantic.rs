use std::cmp::{max, min};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use camino::Utf8PathBuf;
use regex::Regex;

use crate::model::SemanticDiff;
use crate::util::path_norm::redact_run_root;
use crate::util::process::find_on_path;

#[derive(Debug, Clone, Default)]
pub struct SemanticDiffOptions {
    pub diffoscope: Option<Utf8PathBuf>,
    pub no_diffoscope: bool,
    pub left_run_root: Option<PathBuf>,
    pub right_run_root: Option<PathBuf>,
}

pub fn build_semantic_diff(
    left: &Path,
    right: &Path,
    opts: &SemanticDiffOptions,
) -> Result<SemanticDiff> {
    if let Some(diff) = text_diff(left, right)? {
        return Ok(redact(diff, opts));
    }

    if !opts.no_diffoscope {
        if let Some(diff) = diffoscope_diff(left, right, opts)? {
            return Ok(redact(diff, opts));
        }
    }

    if let Some(diff) = tool_fallback_diff(left, right)? {
        return Ok(redact(diff, opts));
    }

    let left_bytes =
        std::fs::read(left).with_context(|| format!("failed to read {}", left.display()))?;
    let right_bytes =
        std::fs::read(right).with_context(|| format!("failed to read {}", right.display()))?;

    let left_strings = printable_strings(&left_bytes);
    let right_strings = printable_strings(&right_bytes);
    if !left_strings.is_empty() || !right_strings.is_empty() {
        let excerpt = summarize_strings_delta(&left_strings, &right_strings);
        let diff = SemanticDiff {
            backend: "strings-fallback".to_string(),
            summary: "binary diff summarized by printable strings".to_string(),
            excerpt: excerpt.clone(),
            left_tokens: tokenize(&excerpt),
            right_tokens: tokenize(&excerpt),
        };
        return Ok(redact(diff, opts));
    }

    let diff = hex_excerpt_diff(&left_bytes, &right_bytes);
    Ok(redact(diff, opts))
}

fn text_diff(left: &Path, right: &Path) -> Result<Option<SemanticDiff>> {
    let Ok(left_text) = std::fs::read_to_string(left) else {
        return Ok(None);
    };
    let Ok(right_text) = std::fs::read_to_string(right) else {
        return Ok(None);
    };

    let left_lines = left_text.lines().collect::<Vec<_>>();
    let right_lines = right_text.lines().collect::<Vec<_>>();
    let total = max(left_lines.len(), right_lines.len());

    let mut excerpt = String::new();
    let mut mismatch = 0usize;
    for idx in 0..total {
        let l = left_lines.get(idx).copied().unwrap_or("");
        let r = right_lines.get(idx).copied().unwrap_or("");
        if l != r {
            mismatch += 1;
            excerpt.push_str(&format!("line {}:\n- {}\n+ {}\n", idx + 1, l, r));
            if mismatch >= 40 {
                excerpt.push_str("... truncated ...\n");
                break;
            }
        }
    }

    if mismatch == 0 {
        excerpt.push_str("text differs only in encoding/newline details or not at all");
    }

    Ok(Some(SemanticDiff {
        backend: "text".to_string(),
        summary: format!("{mismatch} differing lines"),
        excerpt: excerpt.clone(),
        left_tokens: tokenize(&left_text),
        right_tokens: tokenize(&right_text),
    }))
}

fn diffoscope_diff(
    left: &Path,
    right: &Path,
    opts: &SemanticDiffOptions,
) -> Result<Option<SemanticDiff>> {
    let bin = if let Some(explicit) = &opts.diffoscope {
        PathBuf::from(explicit)
    } else if let Some(found) = find_on_path("diffoscope") {
        found
    } else {
        return Ok(None);
    };

    let out = Command::new(bin)
        .arg("--text")
        .arg("-")
        .arg(left)
        .arg(right)
        .output()
        .context("failed to execute diffoscope")?;

    let code = out.status.code().unwrap_or(2);
    if !(code == 0 || code == 1) {
        return Ok(None);
    }

    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    if text.trim().is_empty() {
        text = String::from_utf8_lossy(&out.stderr).into_owned();
    }
    if text.trim().is_empty() {
        return Ok(None);
    }

    let excerpt = text.lines().take(200).collect::<Vec<_>>().join("\n");

    Ok(Some(SemanticDiff {
        backend: "diffoscope".to_string(),
        summary: "diffoscope result (truncated)".to_string(),
        excerpt: excerpt.clone(),
        left_tokens: tokenize(&excerpt),
        right_tokens: tokenize(&excerpt),
    }))
}

fn tool_fallback_diff(left: &Path, right: &Path) -> Result<Option<SemanticDiff>> {
    let tools: [(&str, &[&str]); 4] = [
        ("readelf", &["-Wa"]),
        ("objdump", &["-x", "-s"]),
        ("llvm-dwarfdump", &[]),
        ("strings", &["-a"]),
    ];

    for (tool, args) in tools {
        let Some(bin) = find_on_path(tool) else {
            continue;
        };

        let left_out = Command::new(&bin).args(args).arg(left).output();
        let right_out = Command::new(&bin).args(args).arg(right).output();

        let (Ok(left_out), Ok(right_out)) = (left_out, right_out) else {
            continue;
        };
        if !left_out.status.success() || !right_out.status.success() {
            continue;
        }

        let left_text = String::from_utf8_lossy(&left_out.stdout).into_owned();
        let right_text = String::from_utf8_lossy(&right_out.stdout).into_owned();
        if left_text.is_empty() || right_text.is_empty() || left_text == right_text {
            continue;
        }

        let excerpt = summarize_line_delta(&left_text, &right_text, 80);
        if excerpt.is_empty() {
            continue;
        }

        return Ok(Some(SemanticDiff {
            backend: format!("{tool}-fallback"),
            summary: format!("{tool} output differs"),
            excerpt,
            left_tokens: tokenize(&left_text),
            right_tokens: tokenize(&right_text),
        }));
    }

    Ok(None)
}

fn printable_strings(bytes: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for b in bytes {
        let ch = *b as char;
        if ch.is_ascii_graphic() || ch == ' ' {
            cur.push(ch);
        } else {
            if cur.len() >= 4 {
                out.push(cur.clone());
            }
            cur.clear();
        }
    }
    if cur.len() >= 4 {
        out.push(cur);
    }

    out.truncate(4000);
    out
}

fn summarize_strings_delta(left: &[String], right: &[String]) -> String {
    let mut excerpt = String::new();
    let max_lines = max(left.len(), right.len());
    let mut shown = 0usize;

    for i in 0..max_lines {
        let l = left.get(i).map(String::as_str).unwrap_or("");
        let r = right.get(i).map(String::as_str).unwrap_or("");
        if l != r {
            excerpt.push_str(&format!("- {l}\n+ {r}\n"));
            shown += 1;
            if shown >= 80 {
                excerpt.push_str("... truncated ...\n");
                break;
            }
        }
    }

    if shown == 0 {
        excerpt.push_str("printable strings are identical; binary layout likely differs");
    }
    excerpt
}

fn summarize_line_delta(left: &str, right: &str, limit: usize) -> String {
    let left_lines = left.lines().collect::<Vec<_>>();
    let right_lines = right.lines().collect::<Vec<_>>();
    let total = max(left_lines.len(), right_lines.len());

    let mut excerpt = String::new();
    let mut shown = 0usize;
    for i in 0..total {
        let l = left_lines.get(i).copied().unwrap_or("");
        let r = right_lines.get(i).copied().unwrap_or("");
        if l != r {
            excerpt.push_str(&format!("line {}:\n- {}\n+ {}\n", i + 1, l, r));
            shown += 1;
            if shown >= limit {
                excerpt.push_str("... truncated ...\n");
                break;
            }
        }
    }

    excerpt
}

fn hex_excerpt_diff(left: &[u8], right: &[u8]) -> SemanticDiff {
    let limit = min(left.len(), right.len());
    let mut first_diff = None;
    for i in 0..limit {
        if left[i] != right[i] {
            first_diff = Some(i);
            break;
        }
    }
    let first_diff = first_diff.unwrap_or(limit);
    let from = first_diff.saturating_sub(32);
    let to_left = min(left.len(), first_diff + 96);
    let to_right = min(right.len(), first_diff + 96);

    let left_hex =
        left[from..to_left].iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
    let right_hex =
        right[from..to_right].iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");

    let excerpt = format!(
        "first differing offset: {first_diff}\nleft [{from}..{to_left}]:\n{left_hex}\nright [{from}..{to_right}]:\n{right_hex}\n"
    );

    SemanticDiff {
        backend: "hex-fallback".to_string(),
        summary: "first differing bytes".to_string(),
        excerpt: excerpt.clone(),
        left_tokens: tokenize(&excerpt),
        right_tokens: tokenize(&excerpt),
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let Ok(re) = Regex::new(r"[A-Za-z0-9_./:-]{2,}") else {
        return Vec::new();
    };
    re.find_iter(text).map(|m| m.as_str().to_string()).collect()
}

fn redact(mut diff: SemanticDiff, opts: &SemanticDiffOptions) -> SemanticDiff {
    if let Some(root) = &opts.left_run_root {
        diff.excerpt = redact_run_root(&diff.excerpt, root);
    }
    if let Some(root) = &opts.right_run_root {
        diff.excerpt = redact_run_root(&diff.excerpt, root);
    }
    diff
}

#[cfg(test)]
#[path = "tests/semantic.rs"]
mod tests;
