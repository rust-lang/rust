//! Auto-generate stub docs for the unstable book

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::env;
use std::fs::{self, write};
use std::io::BufReader;
use std::path::Path;
use tidy::features::{collect_lang_features, collect_lib_features, Features};
use tidy::t;
use tidy::unstable_book::{
    collect_unstable_book_section_file_names, collect_unstable_feature_names, LANG_FEATURES_DIR,
    LIB_FEATURES_DIR, PATH_STR,
};

use rustdoc_json_types::{Crate, Id};
use syn::{parse::Parser, Ident, Lit, Meta, NestedMeta};

// (item path, doc url)
fn full_path(krate: &Crate, item: &Id) -> Option<(String, String)> {
    let item_summary = krate.paths.get(item)?;
    let kind = &item_summary.kind;
    let kind_str = serde_json::to_string(kind).ok()?;
    let mut url = String::from("https://doc.rust-lang.org/nightly/");
    let mut iter = item_summary.path.iter();
    iter.next_back();
    url.push_str(&iter.cloned().collect::<Vec<_>>().join("/"));
    url.push('/');
    url.push_str(kind_str.trim_matches('"'));
    url.push('.');
    url.push_str(item_summary.path.last().unwrap());
    url.push_str(".html");
    Some((item_summary.path.join("::"), url))
}

fn is_ident(ident: &Ident, name: &str) -> bool {
    *ident == Ident::new(name, ident.span())
}

/// Returns an unstable feature -> (item path, doc url) mapping.
pub fn load_rustdoc_json_metadata(doc_dir: &Path) -> HashMap<String, Vec<(String, String)>> {
    let mut all_items = HashMap::new();

    for file in fs::read_dir(doc_dir).expect("failed to list files in directory") {
        let entry = file.expect("failed to list file in directory");
        let file = fs::File::open(entry.path()).expect("failed to open file");
        let krate: Crate =
            serde_json::from_reader(BufReader::new(file)).expect("failed to parse JSON docs");

        let mut crate_items = HashMap::new();
        for (id, item) in &krate.index {
            if item.name.is_none() {
                continue;
            }
            let unstable_feature = item.attrs.iter().find_map(|attr: &String| {
                let Ok(parsed) = syn::Attribute::parse_outer.parse_str(attr).map(|mut v| v.swap_remove(0)) else {return None};

                // Make sure this is an `unstable` attribute.
                if !is_ident(parsed.path.get_ident()?, "unstable") {
                    return None;
                }

                // Given `#[unstable(feature = "xyz")]`, return `(feature = "xyz")`.
                let list = match parsed.parse_meta() {
                    Ok(Meta::List(list)) => list,
                    _ => return None,
                };

                // Given a `NestedMeta` like `feature = "xyz"`, returns `xyz`.
                let get_feature_name = |nested: &_| {
                    match nested {
                        NestedMeta::Meta(Meta::NameValue(name_value)) => {
                            if !is_ident(name_value.path.get_ident()?, "feature") {
                                return None;
                            }
                            match &name_value.lit {
                                Lit::Str(s) => Some(s.value()),
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                };

                for nested in list.nested.iter() {
                    if let Some(feat) = get_feature_name(nested) {
                        return Some(feat);
                    }
                }

                None
            });
            if let Some(feat) = unstable_feature {
                crate_items.insert(id, feat);
            }
        }

        for (item, feat) in
            crate_items.into_iter().flat_map(|(item, feat)| Some((full_path(&krate, item)?, feat)))
        {
            all_items.insert(item, feat.to_owned());
        }
    }

    let mut out: HashMap<_, Vec<_>> = HashMap::new();
    for (item, feature) in all_items {
        out.entry(feature).or_default().push(item);
    }

    out
}

fn generate_stub_issue(path: &Path, name: &str, issue: u32, items: &str) {
    let content = format!(include_str!("stub-issue.md"), name = name, issue = issue, items = items);
    t!(write(path, content), path);
}

fn generate_stub_no_issue(path: &Path, name: &str, items: &str) {
    let content = format!(include_str!("stub-no-issue.md"), name = name, items = items);
    t!(write(path, content), path);
}

fn generate_issue(path: &Path, name: &str, issue: u32, items: &str, notes: &str) {
    let content =
        format!(include_str!("issue.md"), name = name, issue = issue, items = items, notes = notes);
    t!(write(path, content), path);
}

fn generate_no_issue(path: &Path, name: &str, items: &str, notes: &str) {
    let content = format!(include_str!("no-issue.md"), name = name, items = items, notes = notes);
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

fn generate_unstable_book_files_lang(src: &Path, out: &Path, features: &Features) {
    let unstable_features = collect_unstable_feature_names(features);
    let unstable_section_file_names = collect_unstable_book_section_file_names(src);
    t!(fs::create_dir_all(&out));
    for feature_name in &unstable_features - &unstable_section_file_names {
        let feature_name_underscore = feature_name.replace('-', "_");
        let file_name = format!("{feature_name}.md");
        let out_file_path = out.join(&file_name);
        let feature = &features[&feature_name_underscore];

        if let Some(issue) = feature.tracking_issue {
            generate_stub_issue(&out_file_path, &feature_name_underscore, issue.get(), "");
        } else {
            generate_stub_no_issue(&out_file_path, &feature_name_underscore, "");
        }
    }
}

fn generate_unstable_book_files_lib(src: &Path, doc: &Path, out: &Path, features: &Features) {
    let unstable_features = collect_unstable_feature_names(features);
    let unstable_section_file_names = collect_unstable_book_section_file_names(src);
    let features_items = load_rustdoc_json_metadata(doc);
    t!(fs::create_dir_all(&out));
    for feature_name in &unstable_features {
        let feature_name_underscore = feature_name.replace('-', "_");
        let file_name = format!("{feature_name}.md");
        let out_file_path = out.join(&file_name);
        let feature = &features[&feature_name_underscore];
        let items = features_items.get(&feature_name_underscore).map(|v| {
            format!(
                "\nItems:\n\n{}",
                v.iter().map(|(path, url)| format!("- [`{path}`]({url})\n")).collect::<String>()
            )
        });
        let notes = if unstable_section_file_names.contains(feature_name) {
            fs::read_to_string(src.join(&file_name)).unwrap()
        } else {
            String::from("")
        };

        if let Some(issue) = feature.tracking_issue {
            generate_issue(
                &out_file_path,
                &feature_name_underscore,
                issue.get(),
                &items.unwrap_or_default(),
                &notes,
            );
        } else {
            generate_no_issue(
                &out_file_path,
                &feature_name_underscore,
                &items.unwrap_or_default(),
                &notes,
            );
        }
    }
}

fn copy_recursive(from: &Path, to: &Path) {
    for entry in t!(fs::read_dir(from)) {
        let e = t!(entry);
        let t = t!(e.metadata());
        let dest = &to.join(e.file_name());
        if t.is_file() {
            if !dest.exists() {
                t!(fs::copy(&e.path(), dest));
            }
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
    let doc_path_str = env::args_os().nth(4).expect("json docs required");
    let dest_path_str = env::args_os().nth(5).expect("destination path required");
    let library_path = Path::new(&library_path_str);
    let compiler_path = Path::new(&compiler_path_str);
    let src_path = Path::new(&src_path_str);
    let doc_path = Path::new(&doc_path_str);
    let dest_path = Path::new(&dest_path_str);

    let lang_features = collect_lang_features(compiler_path, &mut false);
    let lib_features = collect_lib_features(library_path)
        .into_iter()
        .filter(|&(ref name, _)| !lang_features.contains_key(name))
        .collect();

    let doc_src_path = src_path.join(PATH_STR);

    t!(fs::create_dir_all(&dest_path));

    generate_unstable_book_files_lang(
        &doc_src_path.join(LANG_FEATURES_DIR),
        &dest_path.join(LANG_FEATURES_DIR),
        &lang_features,
    );
    generate_unstable_book_files_lib(
        &doc_src_path.join(LIB_FEATURES_DIR),
        doc_path,
        &dest_path.join(LIB_FEATURES_DIR),
        &lib_features,
    );

    copy_recursive(&doc_src_path, &dest_path);

    generate_summary(&dest_path, &lang_features, &lib_features);
}
