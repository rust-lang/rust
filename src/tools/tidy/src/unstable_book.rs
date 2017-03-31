// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashSet;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path;
use features::{collect_lang_features, collect_lib_features, Status};

const PATH_STR: &'static str = "doc/unstable-book/src";

const SUMMARY_FILE_NAME: &'static str = "SUMMARY.md";

static EXCLUDE: &'static [&'static str; 2] = &[SUMMARY_FILE_NAME, "the-unstable-book.md"];

/// Build the path to the Unstable Book source directory from the Rust 'src' directory
fn unstable_book_path(base_src_path: &path::Path) -> path::PathBuf {
    base_src_path.join(PATH_STR)
}

/// Build the path to the Unstable Book SUMMARY file from the Rust 'src' directory
fn unstable_book_summary_path(base_src_path: &path::Path) -> path::PathBuf {
    unstable_book_path(base_src_path).join(SUMMARY_FILE_NAME)
}

/// Open the Unstable Book SUMMARY file
fn open_unstable_book_summary_file(base_src_path: &path::Path) -> fs::File {
    fs::File::open(unstable_book_summary_path(base_src_path))
        .expect("could not open Unstable Book SUMMARY.md")
}

/// Test to determine if DirEntry is a file
fn dir_entry_is_file(dir_entry: &fs::DirEntry) -> bool {
    dir_entry.file_type().expect("could not determine file type of directory entry").is_file()
}

/// Retrieve names of all lang-related unstable features
fn collect_unstable_lang_feature_names(base_src_path: &path::Path) -> HashSet<String> {
    collect_lang_features(base_src_path)
        .into_iter()
        .filter(|&(_, ref f)| f.level == Status::Unstable)
        .map(|(ref name, _)| name.to_owned())
        .collect()
}

/// Retrieve names of all lib-related unstable features
fn collect_unstable_lib_feature_names(base_src_path: &path::Path) -> HashSet<String> {
    let mut bad = true;
    let lang_features = collect_lang_features(base_src_path);
    collect_lib_features(base_src_path, &mut bad, &lang_features)
        .into_iter()
        .filter(|&(_, ref f)| f.level == Status::Unstable)
        .map(|(ref name, _)| name.to_owned())
        .collect()
}

/// Retrieve names of all unstable features
fn collect_unstable_feature_names(base_src_path: &path::Path) -> HashSet<String> {
    collect_unstable_lib_feature_names(base_src_path)
        .union(&collect_unstable_lang_feature_names(base_src_path))
        .map(|n| n.to_owned())
        .collect::<HashSet<_, _>>()
}

/// Retrieve file names of all sections in the Unstable Book with:
///
/// * hyphens replaced by underscores
/// * the markdown suffix ('.md') removed
fn collect_unstable_book_section_file_names(base_src_path: &path::Path) -> HashSet<String> {
    fs::read_dir(unstable_book_path(base_src_path))
        .expect("could not read directory")
        .into_iter()
        .map(|entry| entry.expect("could not read directory entry"))
        .filter(dir_entry_is_file)
        .map(|entry| entry.file_name().into_string().unwrap())
        .filter(|n| EXCLUDE.iter().all(|e| n != e))
        .map(|n| n.trim_right_matches(".md").replace('-', "_"))
        .collect()
}

/// Retrieve unstable feature names that are in the Unstable Book SUMMARY file
fn collect_unstable_book_summary_links(base_src_path: &path::Path) -> HashSet<String> {
    let summary_link_regex =
        ::regex::Regex::new(r"^- \[(\S+)\]\(\S+\.md\)").expect("invalid regex");
    io::BufReader::new(open_unstable_book_summary_file(base_src_path))
        .lines()
        .map(|l| l.expect("could not read line from file"))
        .filter_map(|line| {
            summary_link_regex.captures(&line).map(|c| {
                                                       c.get(1)
                                                           .unwrap()
                                                           .as_str()
                                                           .to_owned()
                                                   })
        })
        .collect()
}

pub fn check(path: &path::Path, bad: &mut bool) {
    let unstable_feature_names = collect_unstable_feature_names(path);
    let unstable_book_section_file_names = collect_unstable_book_section_file_names(path);
    let unstable_book_links = collect_unstable_book_summary_links(path);

    // Check for Unstable Book section names with no corresponding SUMMARY.md link
    for feature_name in &unstable_book_section_file_names - &unstable_book_links {
        *bad = true;
        writeln!(io::stderr(),
                 "The Unstable Book section '{}' needs to have a link in SUMMARY.md",
                 feature_name)
                .expect("could not write to stderr")
    }

    // Check for unstable features that don't have Unstable Book sections
    for feature_name in &unstable_feature_names - &unstable_book_section_file_names {
        *bad = true;
        writeln!(io::stderr(),
                 "Unstable feature '{}' needs to have a section in The Unstable Book",
                 feature_name)
                .expect("could not write to stderr")
    }

    // Check for Unstable Book sections that don't have a corresponding unstable feature
    for feature_name in &unstable_book_section_file_names - &unstable_feature_names {
        *bad = true;
        writeln!(io::stderr(),
                 "The Unstable Book has a section '{}' which doesn't correspond \
                  to an unstable feature",
                 feature_name)
                .expect("could not write to stderr")
    }
}
