use run_make_support::{bin_name, rust_lib_name, rustc};

fn main() {
    rustc().print("crate-name").input("crate.rs").run().assert_stdout_equals("foo");
    rustc().print("file-names").input("crate.rs").run().assert_stdout_equals(bin_name("foo"));
    rustc()
        .print("file-names")
        .crate_type("lib")
        .arg("--test")
        .input("crate.rs")
        .run()
        .assert_stdout_equals(bin_name("foo"));
    rustc()
        .print("file-names")
        .arg("--test")
        .input("lib.rs")
        .run()
        .assert_stdout_equals(bin_name("mylib"));
    rustc().print("file-names").input("lib.rs").run().assert_stdout_equals(rust_lib_name("mylib"));
    rustc().print("file-names").input("rlib.rs").run().assert_stdout_equals(rust_lib_name("mylib"));
}
