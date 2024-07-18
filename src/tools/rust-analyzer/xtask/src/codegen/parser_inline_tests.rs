//! This module greps parser's code for specially formatted comments and turns
//! them into tests.
#![allow(clippy::disallowed_types, clippy::print_stdout)]

use std::{
    collections::HashMap,
    fs, iter,
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::Result;
use itertools::Itertools as _;

use crate::{
    codegen::{ensure_file_contents, reformat, CommentBlock},
    project_root,
    util::list_rust_files,
};

const PARSER_CRATE_ROOT: &str = "crates/parser";
const PARSER_TEST_DATA: &str = "crates/parser/test_data";
const PARSER_TEST_DATA_INLINE: &str = "crates/parser/test_data/parser/inline";

pub(crate) fn generate(check: bool) {
    let tests = tests_from_dir(
        &project_root().join(Path::new(&format!("{PARSER_CRATE_ROOT}/src/grammar"))),
    );

    let mut some_file_was_updated = false;
    some_file_was_updated |=
        install_tests(&tests.ok, &format!("{PARSER_TEST_DATA_INLINE}/ok"), check).unwrap();
    some_file_was_updated |=
        install_tests(&tests.err, &format!("{PARSER_TEST_DATA_INLINE}/err"), check).unwrap();

    if some_file_was_updated {
        let _ = fs::File::open(&format!("{PARSER_CRATE_ROOT}/src/tests.rs"))
            .unwrap()
            .set_modified(SystemTime::now());

        let ok_tests = tests.ok.keys().sorted().map(|k| {
            let test_name = quote::format_ident!("{}", k);
            let test_file = format!("test_data/parser/inline/ok/{test_name}.rs");
            quote::quote! {
                #[test]
                fn #test_name() {
                    run_and_expect_no_errors(#test_file);
                }
            }
        });
        let err_tests = tests.err.keys().sorted().map(|k| {
            let test_name = quote::format_ident!("{}", k);
            let test_file = format!("test_data/parser/inline/err/{test_name}.rs");
            quote::quote! {
                #[test]
                fn #test_name() {
                    run_and_expect_errors(#test_file);
                }
            }
        });

        let output = quote::quote! {
            mod ok {
                use crate::tests::run_and_expect_no_errors;
                #(#ok_tests)*
            }
            mod err {
                use crate::tests::run_and_expect_errors;
                #(#err_tests)*
            }
        };

        let pretty = reformat(output.to_string());
        ensure_file_contents(
            crate::flags::CodegenType::ParserTests,
            format!("{PARSER_TEST_DATA}/generated/runner.rs").as_ref(),
            &pretty,
            check,
        );
    }
}

fn install_tests(tests: &HashMap<String, Test>, into: &str, check: bool) -> Result<bool> {
    let tests_dir = project_root().join(into);
    if !tests_dir.is_dir() {
        fs::create_dir_all(&tests_dir)?;
    }
    let existing = existing_tests(&tests_dir, TestKind::Ok)?;
    if let Some((t, (path, _))) = existing.iter().find(|&(t, _)| !tests.contains_key(t)) {
        panic!("Test `{t}` is deleted: {}", path.display());
    }

    let mut some_file_was_updated = false;

    for (name, test) in tests {
        let path = match existing.get(name) {
            Some((path, _test)) => path.clone(),
            None => tests_dir.join(name).with_extension("rs"),
        };
        if ensure_file_contents(crate::flags::CodegenType::ParserTests, &path, &test.text, check) {
            some_file_was_updated = true;
        }
    }

    Ok(some_file_was_updated)
}

#[derive(Debug)]
struct Test {
    pub name: String,
    pub text: String,
    pub kind: TestKind,
}

#[derive(Copy, Clone, Debug)]
enum TestKind {
    Ok,
    Err,
}

#[derive(Default, Debug)]
struct Tests {
    pub ok: HashMap<String, Test>,
    pub err: HashMap<String, Test>,
}

fn collect_tests(s: &str) -> Vec<Test> {
    let mut res = Vec::new();
    for comment_block in CommentBlock::extract_untagged(s) {
        let first_line = &comment_block.contents[0];
        let (name, kind) = if let Some(name) = first_line.strip_prefix("test ") {
            (name.to_owned(), TestKind::Ok)
        } else if let Some(name) = first_line.strip_prefix("test_err ") {
            (name.to_owned(), TestKind::Err)
        } else {
            continue;
        };
        let text: String = comment_block.contents[1..]
            .iter()
            .cloned()
            .chain(iter::once(String::new()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.trim().is_empty() && text.ends_with('\n'));
        res.push(Test { name, text, kind })
    }
    res
}

fn tests_from_dir(dir: &Path) -> Tests {
    let mut res = Tests::default();
    for entry in list_rust_files(dir) {
        process_file(&mut res, entry.as_path());
    }
    let grammar_rs = dir.parent().unwrap().join("grammar.rs");
    process_file(&mut res, &grammar_rs);
    return res;

    fn process_file(res: &mut Tests, path: &Path) {
        let text = fs::read_to_string(path).unwrap();

        for test in collect_tests(&text) {
            if let TestKind::Ok = test.kind {
                if let Some(old_test) = res.ok.insert(test.name.clone(), test) {
                    panic!("Duplicate test: {}", old_test.name);
                }
            } else if let Some(old_test) = res.err.insert(test.name.clone(), test) {
                panic!("Duplicate test: {}", old_test.name);
            }
        }
    }
}

fn existing_tests(dir: &Path, ok: TestKind) -> Result<HashMap<String, (PathBuf, Test)>> {
    let mut res = HashMap::new();
    for file in fs::read_dir(dir)? {
        let path = file?.path();
        let rust_file = path.extension().and_then(|ext| ext.to_str()) == Some("rs");

        if rust_file {
            let name = path.file_stem().map(|x| x.to_string_lossy().to_string()).unwrap();
            let text = fs::read_to_string(&path)?;
            let test = Test { name: name.clone(), text, kind: ok };
            if let Some(old) = res.insert(name, (path, test)) {
                println!("Duplicate test: {:?}", old);
            }
        }
    }
    Ok(res)
}

#[test]
fn test() {
    generate(true);
}
