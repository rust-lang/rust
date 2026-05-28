//@ only-wasm32-wasip1

use std::collections::{HashMap, HashSet};
use std::path::Path;

use run_make_support::{rfs, rustc, wasmparser};

fn main() {
    test_file("foo.rs", &[("a", &["foo"]), ("b", &["foo"])]);
    test_file("bar.rs", &[("m1", &["f", "g"]), ("m2", &["f"])]);
    test_file("baz.rs", &[("sqlite", &["allocate", "deallocate"])]);
    test_file("log.rs", &[("test", &["log"])]);
}

fn test_file(file: &str, expected_imports: &[(&str, &[&str])]) {
    test(file, &[], expected_imports);
    test(file, &["-Clto"], expected_imports);
    test(file, &["-O"], expected_imports);
    test(file, &["-Clto", "-O"], expected_imports);
}

fn test(file: &str, args: &[&str], expected_imports: &[(&str, &[&str])]) {
    println!("test {file:?} {args:?} for {expected_imports:?}");

    rustc().input(file).target("wasm32-wasip1").args(args).run();

    let file = rfs::read(Path::new(file).with_extension("wasm"));

    let mut imports = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(&file) {
        let payload = payload.unwrap();
        if let wasmparser::Payload::ImportSection(s) = payload {
            for i in s {
                let i = i.unwrap();
                imports.entry(i.module).or_insert(HashSet::new()).insert(i.name);
            }
        }
    }

    eprintln!("imports {imports:?}");

    for (expected_module, expected_names) in expected_imports {
        let names = imports.remove(expected_module).unwrap();
        assert_eq!(names.len(), expected_names.len());
        for name in *expected_names {
            assert!(names.contains(name));
        }
    }
    assert!(imports.is_empty());
}
