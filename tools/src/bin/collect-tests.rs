extern crate file;
extern crate walkdir;
extern crate itertools;

use walkdir::WalkDir;
use itertools::Itertools;

use std::path::{PathBuf, Path};
use std::collections::HashSet;
use std::fs;

fn main() {
    let verify = ::std::env::args().any(|arg| arg == "--verify");

    let d = grammar_dir();
    let tests = tests_from_dir(&d);
    let existing = existing_tests();

    for t in existing.difference(&tests) {
        panic!("Test is deleted: {}\n{}", t.name, t.text);
    }

    let new_tests = tests.difference(&existing);
    for (i, t) in new_tests.enumerate() {
        if verify {
            panic!("Inline test is not recorded: {}", t.name);
        }

        let name = format!("{:04}_{}.rs", existing.len() + i + 1, t.name);
        println!("Creating {}", name);
        let path = inline_tests_dir().join(name);
        file::put_text(&path, &t.text).unwrap();
    }
}


#[derive(Debug, Eq)]
struct Test {
    name: String,
    text: String,
}

impl PartialEq for Test {
    fn eq(&self, other: &Test) -> bool {
        self.name.eq(&other.name)
    }
}

impl ::std::hash::Hash for Test {
    fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

fn tests_from_dir(dir: &Path) -> HashSet<Test> {
    let mut res = HashSet::new();
    for entry in WalkDir::new(dir) {
        let entry = entry.unwrap();
        if !entry.file_type().is_file() {
            continue
        }
        if entry.path().extension().unwrap_or_default() != "rs" {
            continue
        }
        let text = file::get_text(entry.path())
            .unwrap();

        for test in collect_tests(&text) {
            if let Some(old_test) = res.replace(test) {
                panic!("Duplicate test: {}", old_test.name)
            }
        }
    }
    res
}

fn collect_tests(s: &str) -> Vec<Test> {
    let mut res = vec![];
    let prefix = "// ";
    let comment_blocks = s.lines()
        .map(str::trim_left)
        .group_by(|line| line.starts_with(prefix));

    for (is_comment, block) in comment_blocks.into_iter() {
        if !is_comment {
            continue;
        }
        let mut block = block.map(|line| &line[prefix.len()..]);
        let first = block.next().unwrap();
        if !first.starts_with("test ") {
            continue
        }
        let name = first["test ".len()..].to_string();
        let text: String = itertools::join(block.chain(::std::iter::once("")), "\n");
        assert!(!text.trim().is_empty() && text.ends_with("\n"));
        res.push(Test { name, text })
    }
    res
}

fn existing_tests() -> HashSet<Test> {
    let mut res = HashSet::new();
    for file in fs::read_dir(&inline_tests_dir()).unwrap() {
        let file = file.unwrap();
        let path = file.path();
        if path.extension().unwrap_or_default() != "rs" {
            continue
        }
        let name = path.file_name().unwrap().to_str().unwrap();
        let name = name["0000_".len()..name.len() - 3].to_string();
        let text = file::get_text(&path).unwrap();
        res.insert(Test { name, text });
    }
    res
}

fn inline_tests_dir() -> PathBuf {
    let res = base_dir().join("tests/data/parser/inline");
    if !res.is_dir() {
        fs::create_dir_all(&res).unwrap();
    }
    res
}

fn grammar_dir() -> PathBuf {
    base_dir().join("src/parser/event_parser/grammar")
}

fn base_dir() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).parent().unwrap().to_owned()
}


