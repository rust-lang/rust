//! Auto-generate stub docs for the unstable book

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, write};
use std::path::Path;
use tidy::features::{collect_lang_features, collect_lib_features, Features};
use tidy::t;
use tidy::unstable_book::{
    collect_unstable_book_section_file_names, collect_unstable_feature_names, LANG_FEATURES_DIR,
    LIB_FEATURES_DIR, PATH_STR,
};

fn generate_stub_issue(path: &Path, name: &str, issue: u32) {
    let content = format!(include_str!("stub-issue.md"), name = name, issue = issue);
    t!(write(path, content), path);
}

fn generate_stub_no_issue(path: &Path, name: &str) {
    let content = format!(include_str!("stub-no-issue.md"), name = name);
    t!(write(path, content), path);
}

fn set_to_summary_str(set: &BTreeSet<String>, dir: &str) -> String {
    set.iter()
        .map(|ref n| format!("    - [{}]({}/{}.md)", n.replace('-', "_"), dir, n))
        .fold("".to_owned(), |s, a| s + &a + "\n")
}

fn generate_summary(path: &Path, lang_features: &Features, lib_features: &Features) {
    let compiler_flags = collect_unstable_book_section_file_names(&path.join("src/compiler-flags"));

    let compiler_flags_str = set_to_summary_str(&compiler_flags, "compiler-flags");

    let unstable_lang_features = collect_unstable_feature_names(&lang_features);
    let unstable_lib_features = collect_unstable_feature_names(&lib_features);

    let lang_features_str = set_to_summary_str(&unstable_lang_features, "language-features");
    let lib_features_str = set_to_summary_str(&unstable_lib_features, "library-features");

    let summary_path = path.join("src/SUMMARY.md");
    let content = format!(
        include_str!("SUMMARY.md"),
        compiler_flags = compiler_flags_str,
        language_features = lang_features_str,
        library_features = lib_features_str
    );
    t!(write(&summary_path, content), summary_path);
}

fn generate_unstable_book_files(src: &Path, out: &Path, features: &Features) {
    let unstable_features = collect_unstable_feature_names(features);
    let unstable_section_file_names = collect_unstable_book_section_file_names(src);
    t!(fs::create_dir_all(&out));
    for feature_name in &unstable_features - &unstable_section_file_names {
        let feature_name_underscore = feature_name.replace('-', "_");
        let file_name = format!("{feature_name}.md");
        let out_file_path = out.join(&file_name);
        let feature = &features[&feature_name_underscore];

        if let Some(issue) = feature.tracking_issue {
            generate_stub_issue(&out_file_path, &feature_name_underscore, issue.get());
        } else {
            generate_stub_no_issue(&out_file_path, &feature_name_underscore);
        }
    }
}

fn copy_recursive(from: &Path, to: &Path) {
    for entry in t!(fs::read_dir(from)) {
        let e = t!(entry);
        let t = t!(e.metadata());
        let dest = &to.join(e.file_name());
        if t.is_file() {
            t!(fs::copy(&e.path(), dest));
        } else if t.is_dir() {
            t!(fs::create_dir_all(dest));
            copy_recursive(&e.path(), dest);
        }
    }
}

fn main() {
    let library_path_str = env::args_os().nth(1).expect("library/ path required");
    let compiler_path_str = env::args_os().nth(2).expect("compiler/ path required");
    let src_path_str = env::args_os().nth(3).expect("src/ path required");
    let dest_path_str = env::args_os().nth(4).expect("destination path required");
    let library_path = Path::new(&library_path_str);
    let compiler_path = Path::new(&compiler_path_str);
    let src_path = Path::new(&src_path_str);
    let dest_path = Path::new(&dest_path_str);

    let lang_features = collect_lang_features(compiler_path, &mut false);
    let lib_features = collect_lib_features(library_path)
        .into_iter()
        .filter(|&(ref name, _)| !lang_features.contains_key(name))
        .collect();

    let doc_src_path = src_path.join(PATH_STR);

    t!(fs::create_dir_all(&dest_path));

    generate_unstable_book_files(
        &doc_src_path.join(LANG_FEATURES_DIR),
        &dest_path.join(LANG_FEATURES_DIR),
        &lang_features,
    );
    generate_unstable_book_files(
        &doc_src_path.join(LIB_FEATURES_DIR),
        &dest_path.join(LIB_FEATURES_DIR),
        &lib_features,
    );

    copy_recursive(&doc_src_path, &dest_path);

    generate_summary(&dest_path, &lang_features, &lib_features);
}
