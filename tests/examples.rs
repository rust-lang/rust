// FIXME: rustc doesn't generate expansion info for `cfg!` anymore
#![allow(logic_bug)]

#[macro_use]
extern crate duct;

use std::io::{BufRead, BufReader};
use std::fs::File;

#[test]
fn examples() {
    let mut error = false;
    for file in std::fs::read_dir("clippy_tests/examples").unwrap() {
        let file = file.unwrap().path();
        // only test *.rs files
        if file.extension().map_or(true, |file| file != "rs") {
            continue;
        }
        print!("testing {}... ", file.file_stem().unwrap().to_str().unwrap());
        let skip = BufReader::new(File::open(&file).unwrap()).lines().any(|line| {
            let line = line.as_ref().unwrap().trim();
            line == "// ignore-x86" && cfg!(target_pointer_width = "32") ||
            line == "// ignore-x86_64" && cfg!(target_pointer_width = "64")
        });
        if skip {
            println!("skipping");
            continue;
        }
        cmd!("touch", &file).run().unwrap();
        let output = file.with_extension("stderr");
        cmd!("cargo", "rustc", "-q", "--example", file.file_stem().unwrap(), "--", "-Dwarnings",
             "-Zremap-path-prefix-from=examples/", "-Zremap-path-prefix-to=",
             "-Zremap-path-prefix-from=examples\\", "-Zremap-path-prefix-to="
             )
            .unchecked()
            .stderr(&output)
            .env("CLIPPY_DISABLE_WIKI_LINKS", "true")
            .dir("clippy_tests")
            .run()
            .unwrap();
        if cmd!("git", "diff", "--exit-code", output).run().is_err() {
            error = true;
            println!("ERROR");
        } else {
            println!("ok");
        }
    }
    assert!(!error, "A test failed");
}
