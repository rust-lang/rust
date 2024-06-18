// Test that separate compilation actually puts code into separate compilation
// units.  `foo.rs` defines `magic_fn` in three different modules, which should
// wind up in three different compilation units.
// See https://github.com/rust-lang/rust/pull/16367

use run_make_support::{count_regex_matches_in_files_with_extension, regex, rustc};

fn main() {
    rustc().input("foo.rs").emit("llvm-ir").codegen_units(3).run();
    let re = regex::Regex::new(r#"define\ .*magic_fn"#).unwrap();
    assert_eq!(count_regex_matches_in_files_with_extension(&re, "ll"), 3);
}
