//! This module greps parser's code for specially formatted comments and turnes
//! them into tests.

use std::{
    collections::HashMap,
    fs, iter,
    path::{Path, PathBuf},
};

use crate::{
    codegen::{ensure_file_contents, extract_comment_blocks},
    project_root, Result,
};

pub(crate) fn generate_parser_tests() -> Result<()> {
    let tests = tests_from_dir(&project_root().join(Path::new("crates/parser/src/grammar")))?;
    fn install_tests(tests: &HashMap<String, Test>, into: &str) -> Result<()> {
        let tests_dir = project_root().join(into);
        if !tests_dir.is_dir() {
            fs::create_dir_all(&tests_dir)?;
        }
        // ok is never actually read, but it needs to be specified to create a Test in existing_tests
        let existing = existing_tests(&tests_dir, true)?;
        for t in existing.keys().filter(|&t| !tests.contains_key(t)) {
            panic!("Test is deleted: {}", t);
        }

        let mut new_idx = existing.len() + 1;
        for (name, test) in tests {
            let path = match existing.get(name) {
                Some((path, _test)) => path.clone(),
                None => {
                    let file_name = format!("{:04}_{}.rs", new_idx, name);
                    new_idx += 1;
                    tests_dir.join(file_name)
                }
            };
            ensure_file_contents(&path, &test.text)?;
        }
        Ok(())
    }
    install_tests(&tests.ok, "crates/syntax/test_data/parser/inline/ok")?;
    install_tests(&tests.err, "crates/syntax/test_data/parser/inline/err")
}

#[derive(Debug)]
struct Test {
    name: String,
    text: String,
    ok: bool,
}

#[derive(Default, Debug)]
struct Tests {
    ok: HashMap<String, Test>,
    err: HashMap<String, Test>,
}

fn collect_tests(s: &str) -> Vec<Test> {
    let mut res = Vec::new();
    for comment_block in extract_comment_blocks(s) {
        let first_line = &comment_block[0];
        let (name, ok) = if first_line.starts_with("test ") {
            let name = first_line["test ".len()..].to_string();
            (name, true)
        } else if first_line.starts_with("test_err ") {
            let name = first_line["test_err ".len()..].to_string();
            (name, false)
        } else {
            continue;
        };
        let text: String = comment_block[1..]
            .iter()
            .cloned()
            .chain(iter::once(String::new()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.trim().is_empty() && text.ends_with('\n'));
        res.push(Test { name, text, ok })
    }
    res
}

fn tests_from_dir(dir: &Path) -> Result<Tests> {
    let mut res = Tests::default();
    for entry in ::walkdir::WalkDir::new(dir) {
        let entry = entry.unwrap();
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().unwrap_or_default() != "rs" {
            continue;
        }
        process_file(&mut res, entry.path())?;
    }
    let grammar_rs = dir.parent().unwrap().join("grammar.rs");
    process_file(&mut res, &grammar_rs)?;
    return Ok(res);
    fn process_file(res: &mut Tests, path: &Path) -> Result<()> {
        let text = fs::read_to_string(path)?;

        for test in collect_tests(&text) {
            if test.ok {
                if let Some(old_test) = res.ok.insert(test.name.clone(), test) {
                    anyhow::bail!("Duplicate test: {}", old_test.name);
                }
            } else if let Some(old_test) = res.err.insert(test.name.clone(), test) {
                anyhow::bail!("Duplicate test: {}", old_test.name);
            }
        }
        Ok(())
    }
}

fn existing_tests(dir: &Path, ok: bool) -> Result<HashMap<String, (PathBuf, Test)>> {
    let mut res = HashMap::new();
    for file in fs::read_dir(dir)? {
        let file = file?;
        let path = file.path();
        if path.extension().unwrap_or_default() != "rs" {
            continue;
        }
        let name = {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            file_name[5..file_name.len() - 3].to_string()
        };
        let text = xshell::read_file(&path)?;
        let test = Test { name: name.clone(), text, ok };
        if let Some(old) = res.insert(name, (path, test)) {
            println!("Duplicate test: {:?}", old);
        }
    }
    Ok(res)
}
