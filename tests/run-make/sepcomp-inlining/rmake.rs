// Test that #[inline] functions still get inlined across compilation unit
// boundaries. Compilation should produce three IR files, but only the two
// compilation units that have a usage of the #[inline] function should
// contain a definition. Also, the non-#[inline] function should be defined
// in only one compilation unit.
// See https://github.com/rust-lang/rust/pull/16367

use run_make_support::{count_regex_matches_in_file_glob, glob, regex, rustc};

fn main() {
    rustc().input("foo.rs").emit("llvm-ir").codegen_units(3).arg("-Zinline-in-all-cgus").run();
    assert_eq!(count_regex_matches_in_file_glob(r#"define\ i32\ .*inlined"#, "foo.*.ll"), 0);
    assert_eq!(count_regex_matches_in_file_glob(r#"define\ internal\ .*inlined"#, "foo.*.ll"), 2);
    assert_eq!(count_regex_matches_in_file_glob(r#"define\ hidden\ i32\ .*normal"#, "foo.*.ll"), 1);
    assert_eq!(
        count_regex_matches_in_file_glob(r#"declare\ hidden\ i32\ .*normal"#, "foo.*.ll"),
        2
    );
}
