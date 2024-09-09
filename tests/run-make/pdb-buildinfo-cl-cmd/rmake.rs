// Check if the pdb file contains the following information in the LF_BUILDINFO:
// 1. full path to the compiler (cl)
// 2. the commandline args to compile it (cmd)
// This is because these used to be missing in #96475.
// See https://github.com/rust-lang/rust/pull/113492

//@ only-windows-msvc
// Reason: pdb files are unique to this architecture

use run_make_support::{assert_contains, bstr, env_var, rfs, rustc};

fn main() {
    rustc()
        .input("main.rs")
        .arg("-g")
        .crate_name("my_crate_name")
        .crate_type("bin")
        .metadata("dc9ef878b0a48666")
        .run();
    let tests = [
        &env_var("RUSTC"),
        r#""main.rs""#,
        r#""-g""#,
        r#""--crate-name""#,
        r#""my_crate_name""#,
        r#""--crate-type""#,
        r#""bin""#,
        r#""-Cmetadata=dc9ef878b0a48666""#,
    ];
    for test in tests {
        assert_pdb_contains(test);
    }
}

fn assert_pdb_contains(needle: &str) {
    let needle = needle.as_bytes();
    use bstr::ByteSlice;
    assert!(&rfs::read("my_crate_name.pdb").find(needle).is_some());
}
