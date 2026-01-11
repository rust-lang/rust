// This test makes sure that we do not leak paths to the checkout
// (ie. /checkout in CI) in the distributed `libstd` debuginfo.
//
// This test only runs on Linux and dist builder (or with `rust.remap-debuginfo = true`
// set in your `bootstrap.toml`).

//@ needs-std-remap-debuginfo
//@ only-linux

use std::path::PathBuf;

use run_make_support::{llvm_dwarfdump, rfs, rustc, shallow_find_files, source_root};

fn main() {
    // Find the target libdir for the current target
    let target_libdir = {
        let output = rustc().print("target-libdir").run();
        let stdout = output.stdout_utf8();
        let path = PathBuf::from(stdout.trim());

        // Assert that the target-libdir path exists
        assert!(path.exists(), "target-libdir: {path:?} does not exists");

        path
    };

    // Find all the `libstd-.*.rlib` files under the libdir
    let libstd_rlibs = shallow_find_files(&target_libdir, |p| {
        if let Some(filename) = p.file_name()
            && let filename = filename.to_string_lossy()
        {
            filename.starts_with("libstd-") && filename.ends_with(".rlib")
        } else {
            false
        }
    });

    // Assert that there is only one rlib for the `libstd`
    let [libstd_rlib] = &libstd_rlibs[..] else {
        unreachable!("multiple libstd rlib: {libstd_rlibs:?} in {target_libdir:?}");
    };

    // Symlink the libstd rlib here to avoid absolute paths from llvm-dwarfdump own output
    // and not from the debuginfo it-self
    rfs::symlink_file(libstd_rlib, "libstd.rlib");

    // Check that there is only `/rustc/` paths and no `/checkout`, `/home`, or whatever
    llvm_dwarfdump()
        .input("libstd.rlib")
        .run()
        .assert_stdout_contains("/rustc/")
        .assert_stdout_not_contains(source_root().to_string_lossy());
}
