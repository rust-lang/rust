// Test that separate compilation actually puts code into separate compilation
// units.  `foo.rs` defines `magic_fn` in three different modules, which should
// wind up in three different compilation units.
// See https://github.com/rust-lang/rust/pull/16367

use run_make_support::{fs_wrapper, glob, regex, rustc};
use std::io::{BufRead, BufReader};

fn main() {
    let mut match_count = 0;
    rustc().input("foo.rs").emit("llvm-ir").codegen_units(3).run();
    let re = regex::Regex::new(r#"define.*magic_fn"#).unwrap();
    let paths = glob::glob("foo.*.ll").unwrap();
    paths.filter_map(|entry| entry.ok()).filter(|path| path.is_file()).for_each(|path| {
        let file = fs_wrapper::open_file(path);
        let reader = BufReader::new(file);
        reader
            .lines()
            .filter_map(|line| line.ok())
            .filter(|line| re.is_match(line))
            .for_each(|_| match_count += 1);
    });
    assert_eq!(match_count, 3);
}
