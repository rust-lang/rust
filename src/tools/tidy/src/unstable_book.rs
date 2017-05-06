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
use std::path;
use features::{collect_lang_features, collect_lib_features, Status};

const PATH_STR: &'static str = "doc/unstable-book/src";

const LANG_FEATURES_DIR: &'static str = "language-features";

const LIB_FEATURES_DIR: &'static str = "library-features";

/// Build the path to the Unstable Book source directory from the Rust 'src' directory
fn unstable_book_path(base_src_path: &path::Path) -> path::PathBuf {
    base_src_path.join(PATH_STR)
}

/// Directory where the features are documented within the Unstable Book source directory
fn unstable_book_lang_features_path(base_src_path: &path::Path) -> path::PathBuf {
    unstable_book_path(base_src_path).join(LANG_FEATURES_DIR)
}

/// Directory where the features are documented within the Unstable Book source directory
fn unstable_book_lib_features_path(base_src_path: &path::Path) -> path::PathBuf {
    unstable_book_path(base_src_path).join(LIB_FEATURES_DIR)
}

/// Test to determine if DirEntry is a file
fn dir_entry_is_file(dir_entry: &fs::DirEntry) -> bool {
    dir_entry
        .file_type()
        .expect("could not determine file type of directory entry")
        .is_file()
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

fn collect_unstable_book_section_file_names(dir: &path::Path) -> HashSet<String> {
    fs::read_dir(dir)
        .expect("could not read directory")
        .into_iter()
        .map(|entry| entry.expect("could not read directory entry"))
        .filter(dir_entry_is_file)
        .map(|entry| entry.file_name().into_string().unwrap())
        .map(|n| n.trim_right_matches(".md").replace('-', "_"))
        .collect()
}

/// Retrieve file names of all library feature sections in the Unstable Book with:
///
/// * hyphens replaced by underscores
/// * the markdown suffix ('.md') removed
fn collect_unstable_book_lang_features_section_file_names(base_src_path: &path::Path)
                                                          -> HashSet<String> {
    collect_unstable_book_section_file_names(&unstable_book_lang_features_path(base_src_path))
}

/// Retrieve file names of all language feature sections in the Unstable Book with:
///
/// * hyphens replaced by underscores
/// * the markdown suffix ('.md') removed
fn collect_unstable_book_lib_features_section_file_names(base_src_path: &path::Path)
                                                         -> HashSet<String> {
    collect_unstable_book_section_file_names(&unstable_book_lib_features_path(base_src_path))
}

pub fn check(path: &path::Path, bad: &mut bool) {

    // Library features

    let unstable_lib_feature_names = collect_unstable_lib_feature_names(path);
    let unstable_book_lib_features_section_file_names =
        collect_unstable_book_lib_features_section_file_names(path);

    // Check for unstable features that don't have Unstable Book sections
    for feature_name in &unstable_lib_feature_names -
                        &unstable_book_lib_features_section_file_names {
        tidy_error!(bad,
                    "Unstable library feature '{}' needs to have a section within the \
                     'library features' section of The Unstable Book",
                    feature_name);
    }

    // Check for Unstable Book sections that don't have a corresponding unstable feature
    for feature_name in &unstable_book_lib_features_section_file_names -
                        &unstable_lib_feature_names {
        tidy_error!(bad,
                    "The Unstable Book has a 'library feature' section '{}' which doesn't \
                     correspond to an unstable library feature",
                    feature_name)
    }

    // Language features

    let unstable_lang_feature_names = collect_unstable_lang_feature_names(path);
    let unstable_book_lang_features_section_file_names =
        collect_unstable_book_lang_features_section_file_names(path);

    for feature_name in &unstable_lang_feature_names -
                        &unstable_book_lang_features_section_file_names {
        tidy_error!(bad,
                    "Unstable language feature '{}' needs to have a section within the \
                     'language features' section of The Unstable Book",
                    feature_name);
    }

    // Check for Unstable Book sections that don't have a corresponding unstable feature
    for feature_name in &unstable_book_lang_features_section_file_names -
                        &unstable_lang_feature_names {
        tidy_error!(bad,
                    "The Unstable Book has a 'language feature' section '{}' which doesn't \
                     correspond to an unstable language feature",
                    feature_name)
    }
}
