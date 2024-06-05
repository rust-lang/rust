use std::process::Output;

use run_make_support::{bin_name, rust_lib_name, rustc};

fn compare_stdout<S: AsRef<str>>(output: Output, expected: S) {
    assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), expected.as_ref());
}

fn main() {
    compare_stdout(rustc().print("crate-name").input("crate.rs").run(), "foo");
    compare_stdout(rustc().print("file-names").input("crate.rs").run(), bin_name("foo"));
    compare_stdout(
        rustc().print("file-names").crate_type("lib").arg("--test").input("crate.rs").run(),
        bin_name("foo"),
    );
    compare_stdout(
        rustc().print("file-names").arg("--test").input("lib.rs").run(),
        bin_name("mylib"),
    );
    compare_stdout(rustc().print("file-names").input("lib.rs").run(), rust_lib_name("mylib"));
    compare_stdout(rustc().print("file-names").input("rlib.rs").run(), rust_lib_name("mylib"));
}
