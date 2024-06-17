// Check that cross-crate inlined items are inlined in all compilation units
// that refer to them, and not in any other compilation units.
// Note that we have to pass `-C codegen-units=6` because up to two CGUs may be
// created for each source module (see `rustc_const_eval::monomorphize::partitioning`).
// See https://github.com/rust-lang/rust/pull/16367

use run_make_support::{count_regex_matches_in_files_with_extension, regex, rustc};

fn main() {
    rustc().input("cci_lib.rs").run();
    rustc().input("foo.rs").emit("llvm-ir").codegen_units(6).arg("-Zinline-in-all-cgus").run();
    let re = regex::Regex::new(r#"define\ .*cci_fn"#).unwrap();
    assert_eq!(count_regex_matches_in_files_with_extension(&re, "ll"), 2);
}
