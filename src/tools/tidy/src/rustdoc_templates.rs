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
        |path, is_dir| is_dir || path.extension().is_none_or(|ext| ext != OsStr::new("html")),
        &mut |path: &DirEntry, file_content: &str| {
            let mut lines = file_content.lines().enumerate().peekable();

            while let Some((pos, line)) = lines.next() {
                let line = line.trim();
                if let Some(need_next_line_check) = TAGS.iter().find_map(|(tag, end_tag)| {
                    // We first check if the line ends with a jinja tag.
                    if !line.ends_with(end_tag) {
                        None
                    // Then we check if this a comment tag.
                    } else if *tag != "{#" {
                        Some(false)
                    // And finally we check if the comment is empty (ie, only there to strip
                    // extra whitespace characters).
                    } else if let Some(start_pos) = line.rfind(tag) {
                        Some(line[start_pos + 2..].trim() == "#}")
                    } else {
                        Some(false)
                    }
                }) {
                    // All good, the line is ending is a jinja tag. But maybe this tag is useless
                    // if the next line starts with a jinja tag as well!
                    //
                    // However, only (empty) comment jinja tags are concerned about it.
                    if need_next_line_check
                        && lines.peek().is_some_and(|(_, next_line)| {
                            let next_line = next_line.trim_start();
                            TAGS.iter().any(|(tag, _)| next_line.starts_with(tag))
                        })
                    {
                        // It seems like ending this line with a jinja tag is not needed after all.
                        tidy_error!(
                            bad,
                            "`{}` at line {}: unneeded `{{# #}}` tag at the end of the line",
                            path.path().display(),
                            pos + 1,
                        );
                    }
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
