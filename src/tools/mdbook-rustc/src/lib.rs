use std::collections::HashMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::{env, fs};

use mdbook_preprocessor::book::Chapter;

use crate::target::{ByTier, Target};

pub mod target;

const TIER_1_HOST_MARKER: &str = "{{TIER_1_HOST_TABLE}}";
const TIER_1_NOHOST_MARKER: &str = "{{TIER_1_NOHOST_TABLE}}";
const TIER_2_HOST_MARKER: &str = "{{TIER_2_HOST_TABLE}}";
const TIER_2_NOHOST_MARKER: &str = "{{TIER_2_NOHOST_TABLE}}";
const TIER_3_MARKER: &str = "{{TIER_3_TABLE}}";

const EMPTY_TIER_1_NOHOST_MSG: &str =
    "At this time, all Tier 1 targets are [Tier 1 with Host Tools](#tier-1-with-host-tools).\n";
const EMPTY_TIER_2_NOHOST_MSG: &str =
    "At this time, all Tier 2 targets are [Tier 2 with Host Tools](#tier-2-with-host-tools).\n";

pub fn process_main_chapter(
    chapter: &mut Chapter,
    rustc_targets: &[Target],
    target_chapters: &HashMap<String, Chapter>,
) {
    let targets = ByTier::from(rustc_targets);
    let mut new_content = String::new();

    for line in chapter.content.lines() {
        match line.trim() {
            TIER_1_HOST_MARKER => {
                write_host_table(&mut new_content, &targets.tier1_host, &target_chapters)
            }
            TIER_1_NOHOST_MARKER => write_nohost_table(
                &mut new_content,
                &targets.tier1_nohost,
                &target_chapters,
                EMPTY_TIER_1_NOHOST_MSG,
            ),
            TIER_2_HOST_MARKER => {
                write_host_table(&mut new_content, &targets.tier2_host, &target_chapters)
            }
            TIER_2_NOHOST_MARKER => write_nohost_table(
                &mut new_content,
                &targets.tier2_nohost,
                &target_chapters,
                EMPTY_TIER_2_NOHOST_MSG,
            ),
            TIER_3_MARKER => write_tier3_table(&mut new_content, &targets.tier3, &target_chapters),
            _ => {
                new_content.push_str(line);
                new_content.push_str("\n");
            }
        }
    }

    debug_dump("platform-support.md", &new_content);

    chapter.content = new_content;
}

fn write_host_table(
    out: &mut String,
    targets: &[&Target],
    target_chapters: &HashMap<String, Chapter>,
) {
    out.push_str("target | notes\n-------|-------\n");
    for target in targets {
        write_target_tuple(out, target, target_chapters);
        _ = writeln!(out, " | {}", target.metadata.description.as_deref().unwrap_or(""));
    }
}

fn write_nohost_table(
    out: &mut String,
    targets: &[&Target],
    target_chapters: &HashMap<String, Chapter>,
    empty_msg: &str,
) {
    if targets.is_empty() {
        out.push_str(empty_msg);
        return;
    }

    out.push_str("target | std | notes\n-------|:---:|-------\n");
    for target in targets {
        write_target_tuple(out, target, target_chapters);
        _ = writeln!(
            out,
            " | {} | {}",
            std_support_symbol(target.metadata.std),
            target.metadata.description.as_deref().unwrap_or("")
        );
    }
}

fn write_tier3_table(
    out: &mut String,
    targets: &[&Target],
    target_chapters: &HashMap<String, Chapter>,
) {
    out.push_str("target | std | host | notes\n-------|:---:|:----:|-------\n");
    for target in targets {
        write_target_tuple(out, target, target_chapters);
        _ = writeln!(
            out,
            " | {} | {} | {}",
            std_support_symbol(target.metadata.std),
            host_support_symbol(target.metadata.host_tools),
            target.metadata.description.as_deref().unwrap_or("")
        );
    }
}

fn write_target_tuple(
    out: &mut String,
    target: &Target,
    target_chapters: &HashMap<String, Chapter>,
) {
    let doc_chapter = target.metadata.doc_chapter.as_deref().unwrap_or(&*target.tuple);

    if target_chapters.contains_key(doc_chapter) {
        _ = write!(out, "[`{}`](platform-support/{}.md)", target.tuple, doc_chapter);
    } else {
        _ = write!(out, "`{}`", target.tuple);
    }
}

pub fn std_support_symbol(support: Option<bool>) -> &'static str {
    match support {
        Some(true) => "✓",
        Some(false) => "*",
        None => "?",
    }
}

pub fn host_support_symbol(support: Option<bool>) -> &'static str {
    match support {
        Some(true) => "✓",
        Some(false) => "",
        None => "?",
    }
}

pub fn debug_dump<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) {
    if let Some(dir) = env::var_os("MDBOOK_RUSTC_DEBUG_DUMP_DIR")
        && !dir.is_empty()
    {
        let mut dump_path = PathBuf::from(dir);
        dump_path.push(path.as_ref());
        fs::create_dir_all(dump_path.parent().unwrap()).unwrap();
        fs::write(dump_path, contents).unwrap();
    }
}
