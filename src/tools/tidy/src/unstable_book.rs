use std::collections::BTreeSet;
use std::fs;
use std::path;
use features::{collect_lang_features, collect_lib_features, Features, Status};

pub const PATH_STR: &str = "doc/unstable-book/src";

pub const COMPILER_FLAGS_DIR: &str = "compiler-flags";

pub const LANG_FEATURES_DIR: &str = "language-features";

pub const LIB_FEATURES_DIR: &str = "library-features";

/// Build the path to the Unstable Book source directory from the Rust 'src' directory
pub fn unstable_book_path(base_src_path: &path::Path) -> path::PathBuf {
    base_src_path.join(PATH_STR)
}

/// Directory where the features are documented within the Unstable Book source directory
pub fn unstable_book_lang_features_path(base_src_path: &path::Path) -> path::PathBuf {
    unstable_book_path(base_src_path).join(LANG_FEATURES_DIR)
}

/// Directory where the features are documented within the Unstable Book source directory
pub fn unstable_book_lib_features_path(base_src_path: &path::Path) -> path::PathBuf {
    unstable_book_path(base_src_path).join(LIB_FEATURES_DIR)
}

/// Test to determine if DirEntry is a file
fn dir_entry_is_file(dir_entry: &fs::DirEntry) -> bool {
    dir_entry
        .file_type()
        .expect("could not determine file type of directory entry")
        .is_file()
}

/// Retrieve names of all unstable features
pub fn collect_unstable_feature_names(features: &Features) -> BTreeSet<String> {
    features
        .iter()
        .filter(|&(_, ref f)| f.level == Status::Unstable)
        .map(|(name, _)| name.replace('_', "-"))
        .collect()
}

pub fn collect_unstable_book_section_file_names(dir: &path::Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .expect("could not read directory")
        .map(|entry| entry.expect("could not read directory entry"))
        .filter(dir_entry_is_file)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|e| e.to_str().unwrap()) == Some("md"))
        .map(|path| path.file_stem().unwrap().to_str().unwrap().into())
        .collect()
}

/// Retrieve file names of all library feature sections in the Unstable Book with:
///
/// * hyphens replaced by underscores
/// * the markdown suffix ('.md') removed
fn collect_unstable_book_lang_features_section_file_names(base_src_path: &path::Path)
                                                          -> BTreeSet<String> {
    collect_unstable_book_section_file_names(&unstable_book_lang_features_path(base_src_path))
}

/// Retrieve file names of all language feature sections in the Unstable Book with:
///
/// * hyphens replaced by underscores
/// * the markdown suffix ('.md') removed
fn collect_unstable_book_lib_features_section_file_names(base_src_path: &path::Path)
                                                         -> BTreeSet<String> {
    collect_unstable_book_section_file_names(&unstable_book_lib_features_path(base_src_path))
}

pub fn check(path: &path::Path, bad: &mut bool) {

    // Library features

    let lang_features = collect_lang_features(path, bad);
    let lib_features = collect_lib_features(path).into_iter().filter(|&(ref name, _)| {
        !lang_features.contains_key(name)
    }).collect();

    let unstable_lib_feature_names = collect_unstable_feature_names(&lib_features);
    let unstable_book_lib_features_section_file_names =
        collect_unstable_book_lib_features_section_file_names(path);

    // Check for Unstable Book sections that don't have a corresponding unstable feature
    for feature_name in &unstable_book_lib_features_section_file_names -
                        &unstable_lib_feature_names {
        tidy_error!(bad,
                    "The Unstable Book has a 'library feature' section '{}' which doesn't \
                     correspond to an unstable library feature",
                    feature_name)
    }

    // Language features

    let unstable_lang_feature_names = collect_unstable_feature_names(&lang_features);
    let unstable_book_lang_features_section_file_names =
        collect_unstable_book_lang_features_section_file_names(path);

    // Check for Unstable Book sections that don't have a corresponding unstable feature
    for feature_name in &unstable_book_lang_features_section_file_names -
                        &unstable_lang_feature_names {
        tidy_error!(bad,
                    "The Unstable Book has a 'language feature' section '{}' which doesn't \
                     correspond to an unstable language feature",
                    feature_name)
    }

    // List unstable features that don't have Unstable Book sections
    // Remove the comment marker if you want the list printed
    /*
    println!("Lib features without unstable book sections:");
    for feature_name in &unstable_lang_feature_names -
                        &unstable_book_lang_features_section_file_names {
        println!("    * {} {:?}", feature_name, lib_features[&feature_name].tracking_issue);
    }

    println!("Lang features without unstable book sections:");
    for feature_name in &unstable_lib_feature_names-
                        &unstable_book_lib_features_section_file_names {
        println!("    * {} {:?}", feature_name, lang_features[&feature_name].tracking_issue);
    }
    // */
}
