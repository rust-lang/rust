//! This module greps parser's code for specially formatted comments and turnes
//! them into tests.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use crate::{
    codegen::{self, update, Mode},
    project_root, Result,
};

pub fn generate_parser_tests(mode: Mode) -> Result<()> {
    let tests = tests_from_dir(&project_root().join(Path::new(codegen::GRAMMAR_DIR)))?;
    fn install_tests(tests: &HashMap<String, Test>, into: &str, mode: Mode) -> Result<()> {
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
            update(&path, &test.text, mode)?;
        }
        Ok(())
    }
    install_tests(&tests.ok, codegen::OK_INLINE_TESTS_DIR, mode)?;
    install_tests(&tests.err, codegen::ERR_INLINE_TESTS_DIR, mode)
}

#[derive(Debug)]
struct Test {
    pub name: String,
    pub text: String,
    pub ok: bool,
}

#[derive(Default, Debug)]
struct Tests {
    pub ok: HashMap<String, Test>,
    pub err: HashMap<String, Test>,
}

fn collect_tests(s: &str) -> Vec<(usize, Test)> {
    let mut res = vec![];
    let prefix = "// ";
    let lines = s.lines().map(str::trim_start).enumerate();

    let mut block = vec![];
    for (line_idx, line) in lines {
        let is_comment = line.starts_with(prefix);
        if is_comment {
            block.push((line_idx, &line[prefix.len()..]));
        } else {
            process_block(&mut res, &block);
            block.clear();
        }
    }
    process_block(&mut res, &block);
    return res;

    fn process_block(acc: &mut Vec<(usize, Test)>, block: &[(usize, &str)]) {
        if block.is_empty() {
            return;
        }
        let mut ok = true;
        let mut block = block.iter();
        let (start_line, name) = loop {
            match block.next() {
                Some(&(idx, line)) if line.starts_with("test ") => {
                    break (idx, line["test ".len()..].to_string());
                }
                Some(&(idx, line)) if line.starts_with("test_err ") => {
                    ok = false;
                    break (idx, line["test_err ".len()..].to_string());
                }
                Some(_) => (),
                None => return,
            }
        };
        let text: String =
            block.map(|(_, line)| *line).chain(std::iter::once("")).collect::<Vec<_>>().join("\n");
        assert!(!text.trim().is_empty() && text.ends_with('\n'));
        acc.push((start_line, Test { name, text, ok }))
    }
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

        for (_, test) in collect_tests(&text) {
            if test.ok {
                if let Some(old_test) = res.ok.insert(test.name.clone(), test) {
                    Err(format!("Duplicate test: {}", old_test.name))?
                }
            } else {
                if let Some(old_test) = res.err.insert(test.name.clone(), test) {
                    Err(format!("Duplicate test: {}", old_test.name))?
                }
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
        let text = fs::read_to_string(&path)?;
        let test = Test { name: name.clone(), text, ok };
        if let Some(old) = res.insert(name, (path, test)) {
            println!("Duplicate test: {:?}", old);
        }
    }
    Ok(res)
}
