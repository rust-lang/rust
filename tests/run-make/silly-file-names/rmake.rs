// There used to be assert! checks in the compiler to error on encountering
// files starting or ending with < or > respectively, as a preventive measure
// against "fake" files like <anon>. However, this was not truly required,
// as rustc has other checks to verify the veracity of a file. This test includes
// some files with < and > in their names and prints out their output to stdout,
// expecting no errors.
// See https://github.com/rust-lang/rust/issues/73419

//@ ignore-cross-compile
// Reason: the compiled binary is executed
//@ ignore-windows
// Reason: Windows refuses files with < and > in their names

use run_make_support::{diff, rfs, run, rustc};

fn main() {
    rfs::create_file("<leading-lt");
    rfs::write("<leading-lt", r#""comes from a file with a name that begins with <""#);
    rfs::create_file("trailing-gt>");
    rfs::write("trailing-gt>", r#""comes from a file with a name that ends with >""#);
    rustc().input("silly-file-names.rs").output("silly-file-names").run();
    let out = run("silly-file-names").stdout_utf8();
    diff().expected_file("silly-file-names.run.stdout").actual_text("actual-stdout", out).run();
}
