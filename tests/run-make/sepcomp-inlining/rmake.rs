// Test that #[inline] functions still get inlined across compilation unit
// boundaries. Compilation should produce three IR files, but only the two
// compilation units that have a usage of the #[inline] function should
// contain a definition. Also, the non-#[inline] function should be defined
// in only one compilation unit.
// See https://github.com/rust-lang/rust/pull/16367

use run_make_support::{count_regex_matches_in_files_with_extension, regex, rustc};

fn main() {
    rustc().input("foo.rs").emit("llvm-ir").codegen_units(3).arg("-Zinline-in-all-cgus").run();
    let re = regex::Regex::new(r#"define\ i32\ .*inlined"#).unwrap();
    assert_eq!(count_regex_matches_in_files_with_extension(&re, "ll"), 0);
    let re = regex::Regex::new(r#"define\ internal\ .*inlined"#).unwrap();
    assert_eq!(count_regex_matches_in_files_with_extension(&re, "ll"), 2);
    let re = regex::Regex::new(r#"define\ hidden\ i32\ .*normal"#).unwrap();
    assert_eq!(count_regex_matches_in_files_with_extension(&re, "ll"), 1);
    let re = regex::Regex::new(r#"declare\ hidden\ i32\ .*normal"#).unwrap();
    assert_eq!(count_regex_matches_in_files_with_extension(&re, "ll"), 2);
}
