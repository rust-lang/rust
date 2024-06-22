use run_make_support::fs_wrapper::read_to_string;
use run_make_support::regex::Regex;
use run_make_support::{read_dir, rustc};

use std::ffi::OsStr;

fn main() {
    rustc().input("foo.rs").emit("llvm-ir").codegen_units(2).run();
    let re = Regex::new(r"\bcall\b").unwrap();
    let mut nb_ll = 0;
    read_dir(".", |path| {
        if path.is_file() && path.extension().is_some_and(|ext| ext == OsStr::new("ll")) {
            assert!(!re.is_match(&read_to_string(path)));
            nb_ll += 1;
        }
    });
    assert!(nb_ll > 0);
}
