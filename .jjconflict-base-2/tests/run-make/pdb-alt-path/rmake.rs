// The information inside a .exe file contains a string of the PDB file name.
// This could be a security concern if the full path was exposed, as it could
// reveal information about the filesystem where the bin was first compiled.
// This should only be overridden by `-Clink-arg=/PDBALTPATH:...` - this test
// checks that no full file paths are exposed and that the override flag is respected.
// See https://github.com/rust-lang/rust/pull/121297

//@ only-x86_64-pc-windows-msvc

use run_make_support::{bin_name, invalid_utf8_contains, invalid_utf8_not_contains, run, rustc};

fn main() {
    // Test that we don't have the full path to the PDB file in the binary
    rustc()
        .input("main.rs")
        .arg("-g")
        .crate_name("my_crate_name")
        .crate_type("bin")
        .arg("-Cforce-frame-pointers")
        .run();
    invalid_utf8_contains(&bin_name("my_crate_name"), "my_crate_name.pdb");
    invalid_utf8_not_contains(&bin_name("my_crate_name"), r#"\my_crate_name.pdb"#);
    // Test that backtraces still can find debuginfo by checking that they contain symbol names and
    // source locations.
    let out = run(&bin_name("my_crate_name"));
    out.assert_stdout_contains("my_crate_name::fn_in_backtrace");
    out.assert_stdout_contains("main.rs:15");
    // Test that explicitly passed `-Clink-arg=/PDBALTPATH:...` is respected
    rustc()
        .input("main.rs")
        .arg("-g")
        .crate_name("my_crate_name")
        .crate_type("bin")
        .link_arg("/PDBALTPATH:abcdefg.pdb")
        .arg("-Cforce-frame-pointers")
        .run();
    invalid_utf8_contains(&bin_name("my_crate_name"), "abcdefg.pdb");
    invalid_utf8_not_contains(&bin_name("my_crate_name"), "my_crate_name.pdb");
}
