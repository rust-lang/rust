//! Checks that the rustdoc ID map is up-to-date. The goal here is to check a few things:
//!
//! * All IDs created by rustdoc (through JS or files generation) are declared in the ID map.
//! * There are no unused IDs.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use regex::Regex;

const ID_MAP_PATH: &str = "librustdoc/html/markdown.rs";
const IDS_USED_IN_JS: &[&str] = &[
    // This one is created in the JS cannot be found in rust files.
    "help",
];

fn extract_ids(path: &Path, bad: &mut bool) -> HashMap<String, usize> {
    let file = File::open(path).expect("failed to open file to extract rustdoc IDs");
    let buf_reader = BufReader::new(file);
    let mut iter = buf_reader.lines();
    let mut ids = HashMap::new();

    while let Some(Ok(line)) = iter.next() {
        if line.starts_with("fn init_id_map() -> ") {
            break;
        }
    }
    // We're now in the function body, time to retrieve the IDs!
    while let Some(Ok(line)) = iter.next() {
        if line.trim_start().starts_with("map.insert(\"") {
            let id = line.split('"').skip(1).next().unwrap();
            if ids.insert(id.to_owned(), 0).is_some() {
                eprintln!(
                    "=> ID `{}` is defined more than once in the ID map in file `{}`",
                    id, ID_MAP_PATH
                );
                *bad = true;
            }
        } else if line == "}" {
            // We reached the end of the function.
            break;
        }
    }
    if ids.is_empty() {
        eprintln!("=> No IDs were found in rustdoc in file `{}`...", ID_MAP_PATH);
        *bad = true;
    }
    ids
}

fn check_id(
    path: &Path,
    id: &str,
    ids: &mut HashMap<String, usize>,
    line_nb: usize,
    bad: &mut bool,
) {
    if id.contains('{') {
        // This is a formatted ID, no need to check it!
        return;
    }
    let id = id.to_owned();
    match ids.get_mut(&id) {
        Some(nb) => *nb += 1,
        None => {
            eprintln!(
                "=> ID `{}` in file `{}` at line {} is missing from `init_id_map`",
                id,
                path.display(),
                line_nb + 1,
            );
            *bad = true;
        }
    }
}

fn check_ids(
    path: &Path,
    f: &str,
    ids: &mut HashMap<String, usize>,
    regex: &Regex,
    bad: &mut bool,
) {
    let mut is_checking_small_section_header = None;

    for (line_nb, line) in f.lines().enumerate() {
        let trimmed = line.trim_start();
        // We're not interested in comments or doc comments.
        if trimmed.starts_with("//") {
            continue;
        } else if let Some(start_line) = is_checking_small_section_header {
            if line_nb == start_line + 2 {
                check_id(path, trimmed.split('"').skip(1).next().unwrap(), ids, line_nb, bad);
                is_checking_small_section_header = None;
            }
        } else if trimmed.starts_with("write_small_section_header(") {
            // This is a corner case: the second argument of the function is an ID and we need to
            // check it as well.
            if trimmed.contains(',') {
                // This is a call made on one line, so we can simply check it!
                check_id(path, trimmed.split('"').skip(1).next().unwrap(), ids, line_nb, bad);
            } else {
                is_checking_small_section_header = Some(line_nb);
            }
            continue;
        }
        for cap in regex.captures_iter(line) {
            check_id(path, &cap[1], ids, line_nb, bad);
        }
    }
}

pub fn check(path: &Path, bad: &mut bool) {
    // matches ` id="blabla"`
    let regex = Regex::new(r#"[\s"]id=\\?["']([^\s\\]+)\\?["'][\s\\>"{]"#).unwrap();

    println!("Checking rustdoc IDs...");
    let mut ids = extract_ids(&path.join(ID_MAP_PATH), bad);
    if *bad {
        return;
    }
    super::walk(
        &path.join("librustdoc/html"),
        &mut |path| super::filter_dirs(path),
        &mut |entry, contents| {
            let path = entry.path();
            let file_name = entry.file_name();
            if path.extension() == Some(OsStr::new("html"))
                || (path.extension() == Some(OsStr::new("rs")) && file_name != "tests.rs")
            {
                check_ids(path, contents, &mut ids, &regex, bad);
            }
        },
    );
    for (id, nb) in ids {
        if IDS_USED_IN_JS.iter().any(|i| i == &id) {
            if nb != 0 {
                eprintln!("=> ID `{}` is not supposed to be used in Rust code but in the JS!", id);
                *bad = true;
            }
        } else if nb == 0 {
            eprintln!(
                "=> ID `{}` is unused, it should be removed from `init_id_map` in file `{}`",
                id, ID_MAP_PATH
            );
            *bad = true;
        }
    }
}
