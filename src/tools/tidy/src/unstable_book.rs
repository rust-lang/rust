use std::collections::BTreeSet;
use std::fs;
use std::path::{PathBuf, Path};
use crate::features::{CollectedFeatures, Features, Feature, Status};

pub const PATH_STR: &str = "doc/unstable-book";

pub const COMPILER_FLAGS_DIR: &str = "src/compiler-flags";

pub const LANG_FEATURES_DIR: &str = "src/language-features";

pub const LIB_FEATURES_DIR: &str = "src/library-features";

/// Builds the path to the Unstable Book source directory from the Rust 'src' directory.
pub fn unstable_book_path(base_src_path: &Path) -> PathBuf {
    base_src_path.join(PATH_STR)
}

/// Builds the path to the directory where the features are documented within the Unstable Book
/// source directory.
pub fn unstable_book_lang_features_path(base_src_path: &Path) -> PathBuf {
    unstable_book_path(base_src_path).join(LANG_FEATURES_DIR)
}

/// Builds the path to the directory where the features are documented within the Unstable Book
/// source directory.
pub fn unstable_book_lib_features_path(base_src_path: &Path) -> PathBuf {
    unstable_book_path(base_src_path).join(LIB_FEATURES_DIR)
}

/// Tests whether `DirEntry` is a file.
fn dir_entry_is_file(dir_entry: &fs::DirEntry) -> bool {
    dir_entry
        .file_type()
        .expect("could not determine file type of directory entry")
        .is_file()
}

/// Retrieves names of all unstable features.
pub fn collect_unstable_feature_names(features: &Features) -> BTreeSet<String> {
    features
        .iter()
        .filter(|&(_, ref f)| f.level == Status::Unstable)
        .map(|(name, _)| name.replace('_', "-"))
        .collect()
}

pub fn collect_unstable_book_section_file_names(dir: &Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .expect("could not read directory")
        .map(|entry| entry.expect("could not read directory entry"))
        .filter(dir_entry_is_file)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|e| e.to_str().unwrap()) == Some("md"))
        .map(|path| path.file_stem().unwrap().to_str().unwrap().into())
        .collect()
}

/// Retrieves file names of all library feature sections in the Unstable Book with:
///
/// * hyphens replaced by underscores,
/// * the markdown suffix ('.md') removed.
fn collect_unstable_book_lang_features_section_file_names(base_src_path: &Path)
                                                          -> BTreeSet<String> {
    collect_unstable_book_section_file_names(&unstable_book_lang_features_path(base_src_path))
}

/// Retrieves file names of all language feature sections in the Unstable Book with:
///
/// * hyphens replaced by underscores,
/// * the markdown suffix ('.md') removed.
fn collect_unstable_book_lib_features_section_file_names(base_src_path: &Path) -> BTreeSet<String> {
    collect_unstable_book_section_file_names(&unstable_book_lib_features_path(base_src_path))
}

pub fn check(path: &Path, features: CollectedFeatures, bad: &mut bool) {
    let lang_features = features.lang;
    let mut lib_features = features.lib.into_iter().filter(|&(ref name, _)| {
        !lang_features.contains_key(name)
    }).collect::<Features>();

    // This library feature is defined in the `compiler_builtins` crate, which
    // has been moved out-of-tree. Now it can no longer be auto-discovered by
    // `tidy`, because we need to filter out its (submodule) directory. Manually
    // add it to the set of known library features so we can still generate docs.
    lib_features.insert("compiler_builtins_lib".to_owned(), Feature {
        level: Status::Unstable,
        since: None,
        has_gate_test: false,
        tracking_issue: None,
    });

    // Library features
    let unstable_lib_feature_names = collect_unstable_feature_names(&lib_features);
    let unstable_book_lib_features_section_file_names =
        collect_unstable_book_lib_features_section_file_names(path);

    // Language features
    let unstable_lang_feature_names = collect_unstable_feature_names(&lang_features);
    let unstable_book_lang_features_section_file_names =
        collect_unstable_book_lang_features_section_file_names(path);

    // Check for Unstable Book sections that don't have a corresponding unstable feature
    for feature_name in &unstable_book_lib_features_section_file_names -
                        &unstable_lib_feature_names {
        if !unstable_lang_feature_names.contains(&feature_name) {
            tidy_error!(bad,
                        "The Unstable Book has a 'library feature' section '{}' which doesn't \
                         correspond to an unstable library feature",
                        feature_name);
        }
    }

    // Check for Unstable Book sections that don't have a corresponding unstable feature.
    for feature_name in &unstable_book_lang_features_section_file_names -
                        &unstable_lang_feature_names {
        tidy_error!(bad,
                    "The Unstable Book has a 'language feature' section '{}' which doesn't \
                     correspond to an unstable language feature",
                    feature_name)
    }

    // List unstable features that don't have Unstable Book sections.
    // Remove the comment marker if you want the list printed.
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
