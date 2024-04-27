//! Checks that all Fluent messages appear at least twice

use crate::walk::{filter_dirs, walk};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;

lazy_static::lazy_static! {
    static ref WORD: Regex = Regex::new(r"\w+").unwrap();
}

fn filter_used_messages(
    contents: &str,
    msgs_not_appeared_yet: &mut HashMap<String, String>,
    msgs_appeared_only_once: &mut HashMap<String, String>,
) {
    // we don't just check messages never appear in Rust files,
    // because messages can be used as parts of other fluent messages in Fluent files,
    // so we do checking messages appear only once in all Rust and Fluent files.
    let mut matches = WORD.find_iter(contents);
    while let Some(name) = matches.next() {
        if let Some((name, filename)) = msgs_not_appeared_yet.remove_entry(name.as_str()) {
            // if one msg appears for the first time,
            // remove it from `msgs_not_appeared_yet` and insert it into `msgs_appeared_only_once`.
            msgs_appeared_only_once.insert(name, filename);
        } else {
            // if one msg appears for the second time,
            // remove it from `msgs_appeared_only_once`.
            msgs_appeared_only_once.remove(name.as_str());
        }
    }
}

pub fn check(path: &Path, mut all_defined_msgs: HashMap<String, String>, bad: &mut bool) {
    let mut msgs_appear_only_once = HashMap::new();
    walk(path, |path, _| filter_dirs(path), &mut |_, contents| {
        filter_used_messages(contents, &mut all_defined_msgs, &mut msgs_appear_only_once);
    });

    for (name, filename) in msgs_appear_only_once {
        tidy_error!(bad, "{filename}: message `{}` is not used", name,);
    }
}
