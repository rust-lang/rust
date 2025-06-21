//! Auto-generate stub docs for the unstable book

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, write};
use std::path::Path;
use std::process::Command;

use tidy::features::{Feature, Features, Status, collect_lang_features, collect_lib_features};
use tidy::t;
use tidy::unstable_book::{
    COMPILER_FLAGS_DIR, LANG_FEATURES_DIR, LIB_FEATURES_DIR, PATH_STR,
    collect_unstable_book_section_file_names, collect_unstable_feature_names,
};

fn generate_stub_issue(path: &Path, name: &str, issue: u32, description: &str) {
    let content = format!(
        include_str!("stub-issue.md"),
        name = name,
        issue = issue,
        description = description
    );
    t!(write(path, content), path);
}

fn generate_stub_no_issue(path: &Path, name: &str, description: &str) {
    let content = format!(include_str!("stub-no-issue.md"), name = name, description = description);
    t!(write(path, content), path);
}

fn set_to_summary_str(set: &BTreeSet<String>, dir: &str) -> String {
    set.iter()
        .map(|ref n| format!("    - [{}]({}/{}.md)", n.replace('-', "_"), dir, n))
        .fold("".to_owned(), |s, a| s + &a + "\n")
}

fn generate_summary(
    path: &Path,
    lang_features: &Features,
    lib_features: &Features,
    compiler_flags: &Features,
) {
    let compiler_flags =
        &collect_unstable_book_section_file_names(&path.join("src/compiler-flags"))
            | &collect_unstable_feature_names(&compiler_flags);
    let compiler_env_vars =
        collect_unstable_book_section_file_names(&path.join("src/compiler-environment-variables"));

    let compiler_flags_str = set_to_summary_str(&compiler_flags, "compiler-flags");
    let compiler_env_vars_str =
        set_to_summary_str(&compiler_env_vars, "compiler-environment-variables");

    let unstable_lang_features = collect_unstable_feature_names(&lang_features);
    let unstable_lib_features = collect_unstable_feature_names(&lib_features);

    let lang_features_str = set_to_summary_str(&unstable_lang_features, "language-features");
    let lib_features_str = set_to_summary_str(&unstable_lib_features, "library-features");

    let summary_path = path.join("src/SUMMARY.md");
    let content = format!(
        include_str!("SUMMARY.md"),
        compiler_env_vars = compiler_env_vars_str,
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
        let description = feature.description.as_deref().unwrap_or_default();

        if let Some(issue) = feature.tracking_issue {
            generate_stub_issue(
                &out_file_path,
                &feature_name_underscore,
                issue.get(),
                &description,
            );
        } else {
            generate_stub_no_issue(&out_file_path, &feature_name_underscore, &description);
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

fn collect_compiler_flags(rustc_path: impl AsRef<Path>) -> Features {
    let mut rustc = Command::new(rustc_path.as_ref());
    rustc.arg("-Zhelp");

    let output = t!(rustc.output());
    let help_str = t!(String::from_utf8(output.stdout));
    let parts = help_str.split("\n    -Z").collect::<Vec<_>>();

    let mut features = Features::new();
    for part in parts.into_iter().skip(1) {
        let (name, description) =
            part.split_once("--").expect("name and description should be delimited by '--'");
        let name = name.trim().trim_end_matches("=val");
        let description = description.trim();

        features.insert(
            name.replace('-', "_"),
            Feature {
                level: Status::Unstable,
                since: None,
                has_gate_test: false,
                tracking_issue: None,
                file: "".into(),
                line: 0,
                description: Some(description.to_owned()),
            },
        );
    }
    features
}

fn main() {
    let library_path_str = env::args_os().nth(1).expect("library/ path required");
    let compiler_path_str = env::args_os().nth(2).expect("compiler/ path required");
    let src_path_str = env::args_os().nth(3).expect("src/ path required");
    let rustc_path_str = env::args_os().nth(4).expect("rustc path required");
    let dest_path_str = env::args_os().nth(5).expect("destination path required");
    let library_path = Path::new(&library_path_str);
    let compiler_path = Path::new(&compiler_path_str);
    let src_path = Path::new(&src_path_str);
    let rustc_path = Path::new(&rustc_path_str);
    let dest_path = Path::new(&dest_path_str);

    let lang_features = collect_lang_features(compiler_path, &mut false);
    let lib_features = collect_lib_features(library_path)
        .into_iter()
        .filter(|&(ref name, _)| !lang_features.contains_key(name))
        .collect();
    let compiler_flags = collect_compiler_flags(rustc_path);

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
    generate_unstable_book_files(
        &doc_src_path.join(COMPILER_FLAGS_DIR),
        &dest_path.join(COMPILER_FLAGS_DIR),
        &compiler_flags,
    );

    copy_recursive(&doc_src_path, &dest_path);

    generate_summary(&dest_path, &lang_features, &lib_features, &compiler_flags);
}
