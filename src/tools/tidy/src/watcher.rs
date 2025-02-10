//! Checks that text between tags unchanged, emitting warning otherwise,
//! allowing asserting that code in different places over codebase is in sync.
//!
//! This works via hashing text between tags and saving hash in tidy.
//!
//! Usage:
//!
//! some.rs:
//! // tidy-ticket-foo
//! const FOO: usize = 42;
//! // tidy-ticket-foo
//!
//! some.sh:
//! # tidy-ticket-foo
//! export FOO=42
//! # tidy-ticket-foo
use std::fs;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};
use serde_json;

#[cfg(test)]
mod tests;

#[derive(Deserialize, Serialize, Debug)]
struct TagGroups {
    tag_groups: Vec<TagGroup>,
}

#[derive(Deserialize, Serialize, Debug)]
struct TagGroup {
    name: String,
    #[serde(skip_serializing_if = "is_false")]
    #[serde(default)]
    /// if group sync in broken but you don't want to remove it
    is_off: bool,
    tags: Vec<Tag>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Tag {
    /// path to file
    path: PathBuf,
    /// md5 tag of tag content
    hash: String,
    /// tag string
    tag: String,
}

fn is_false(b: &bool) -> bool {
    !b
}

/// Return hash for source text between 2 tag occurrence,
/// ignoring lines where tag written
///
/// Expecting:
/// tag is not multiline
/// source always have at least 2 occurrence of tag (>2 ignored)
fn span_hash(source: &str, tag: &str, bad: &mut bool, file_path: &Path) -> Result<String, ()> {
    let start_idx = match source.find(tag) {
        Some(idx) => idx,
        None => {
            return Err(tidy_error!(
                bad,
                "tag {} should exist in file {}",
                tag,
                file_path.display()
            ));
        }
    };
    let end_idx = {
        let end = match source[start_idx + tag.len()..].find(tag) {
            // index from source start
            Some(idx) => start_idx + tag.len() + idx,
            None => return Err(tidy_error!(bad, "tag end {} should exist in provided text", tag)),
        };
        // second line with tag can contain some other text before tag, ignore it
        // by finding position of previous line ending
        //
        // FIXME: what if line ending is \r\n? In that case \r will be hashed too
        let offset = source[start_idx..end].rfind('\n').unwrap();
        start_idx + offset
    };

    let mut hasher = Md5::new();

    source[start_idx..end_idx]
        .lines()
        // skip first line with tag
        .skip(1)
        // hash next lines, ignoring end trailing whitespaces
        .for_each(|line| {
            let trimmed = line.trim_end();
            hasher.update(trimmed);
        });
    Ok(format!("{:x}", hasher.finalize()))
}

fn check_entry(
    entry: &mut Tag,
    tag_group_name: &str,
    bad: &mut bool,
    root_path: &Path,
    bless: bool,
) {
    let file_path = root_path.join(Path::new(&entry.path));
    let file = fs::read_to_string(&file_path)
        .unwrap_or_else(|e| panic!("{:?}, path: {}", e, entry.path.display()));
    let actual_hash = span_hash(&file, &entry.tag, bad, &file_path).unwrap();
    if actual_hash != entry.hash {
        if !bless {
            // Write tidy error description for watcher only once.
            // Will not work if there was previous errors of other types.
            if *bad == false {
                tidy_error!(
                    bad,
                    "The code blocks tagged with tidy watcher has changed.\n\
                 It's likely that code blocks with the following tags need to be changed too. Check src/tools/tidy/src/watcher.rs, find tag/hash in TIDY_WATCH_LIST list \
                and verify that sources for provided group of tags in sync. Once that done, run tidy again and update hashes in TIDY_WATCH_LIST with provided actual hashes."
                )
            }
            tidy_error!(
                bad,
                "hash for tag `{}` in path `{}` mismatch:\n  actual: `{}`, expected: `{}`\n  \
            Verify that tags from tag_group `{}` in sync.",
                entry.tag,
                entry.path.display(),
                actual_hash,
                entry.hash,
                tag_group_name,
            );
        } else {
            entry.hash = actual_hash;
        }
    }
}

/*
    // sync self-profile-events help mesage with actual list of events
    add_group!(
        ("compiler/rustc_data_structures/src/profiling.rs", "881e7899c7d6904af1bc000594ee0418", "tidy-ticket-self-profile-events"),
        ("compiler/rustc_session/src/options.rs", "012ee5a3b61ee1377744e5c6913fa00a", "tidy-ticket-self-profile-events")
    ),

    // desynced, pieces in compiler/rustc_pattern_analysis/src/rustc.rs
    // add_group!(
    //     ("compiler/rustc_pattern_analysis/src/constructor.rs", "c17706947fc814aa5648972a5b3dc143", "tidy-ticket-arity"),
    //     // ("compiler/rustc_mir_build/src/thir/pattern/deconstruct_pat.rs", "7ce77b84c142c22530b047703ef209f0", "tidy-ticket-wildcards")
    // ),

    // desynced, pieces in compiler/rustc_hir_analysis/src/lib.rs missing?
    //add_group!( // bad
    //    ("compiler/rustc_hir_analysis/src/lib.rs", "842e23fb65caf3a96681686131093316", "tidy-ticket-sess-time-item_types_checking"),
    //    ("src/librustdoc/core.rs", "85d9dd0cbb94fd521e2d15a8ed38a75f", "tidy-ticket-sess-time-item_types_checking")
    // ),
*/
pub fn check(root_path: &Path, bless: bool, bad: &mut bool) {
    let config_path = root_path.join(Path::new("src/tools/tidy/src/watcher_list.json"));
    let config_file = fs::read_to_string(&config_path).unwrap_or_else(|e| panic!("{:?}", e));
    let mut tag_groups: TagGroups = serde_json::from_str(&config_file).unwrap();
    for tag_group in tag_groups.tag_groups.iter_mut() {
        if !tag_group.is_off {
            for entry in tag_group.tags.iter_mut() {
                check_entry(entry, &tag_group.name, bad, root_path, bless);
            }
        }
    }
    if bless {
        let f = OpenOptions::new().write(true).truncate(true).open(config_path).unwrap();
        serde_json::to_writer_pretty(f, &tag_groups).unwrap();
    }
}
