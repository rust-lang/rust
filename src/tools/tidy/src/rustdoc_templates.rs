//! Tidy check to ensure that rustdoc templates didn't forget a `{# #}` to strip extra whitespace
//! characters.

use std::ffi::OsStr;
use std::path::Path;

use ignore::DirEntry;

use crate::walk::walk;

// Array containing `("beginning of tag", "end of tag")`.
const TAGS: &[(&str, &str)] = &[("{#", "#}"), ("{%", "%}"), ("{{", "}}")];

pub fn check(librustdoc_path: &Path, bad: &mut bool) {
    walk(
        &librustdoc_path.join("html/templates"),
        |path, is_dir| is_dir || !path.extension().is_some_and(|ext| ext == OsStr::new("html")),
        &mut |path: &DirEntry, file_content: &str| {
            let mut lines = file_content.lines().enumerate().peekable();

            while let Some((pos, line)) = lines.next() {
                let line = line.trim();
                if TAGS.iter().any(|(_, tag)| line.ends_with(tag)) {
                    continue;
                }
                let Some(next_line) = lines.peek().map(|(_, next_line)| next_line.trim()) else {
                    continue;
                };
                if TAGS.iter().any(|(tag, _)| next_line.starts_with(tag)) {
                    continue;
                }
                // Maybe this is a multi-line tag, let's filter it out then.
                match TAGS.iter().find_map(|(tag, end_tag)| {
                    if line.rfind(tag).is_some() { Some(end_tag) } else { None }
                }) {
                    None => {
                        // No it's not, let's error.
                        tidy_error!(
                            bad,
                            "`{}` at line {}: missing `{{# #}}` at the end of the line",
                            path.path().display(),
                            pos + 1,
                        );
                    }
                    Some(end_tag) => {
                        // We skip the tag.
                        while let Some((_, next_line)) = lines.peek() {
                            if next_line.contains(end_tag) {
                                break;
                            }
                            lines.next();
                        }
                    }
                }
            }
        },
    );
}
