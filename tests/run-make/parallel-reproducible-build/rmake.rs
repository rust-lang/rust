//@ ignore-windows-gnu
// GNU Linker for Windows is non-deterministic. (from `reproducible-build-2` test in this suite)

use std::rc::Rc;

use run_make_support::{bin_name, is_windows_msvc, rfs, run_in_tmpdir, rustc};

/// Test that parallel compiler produces identical binaries.
fn main() {
    const FILE_NAME: &str = "static-muts-issue-140413";
    let bin_name = bin_name(FILE_NAME);

    let mut reference = None;

    for _ in 0..100 {
        // Tmp dir as previous runs affect output binary on windows.
        run_in_tmpdir(|| {
            let mut rustc = rustc();
            rustc.input(format!("{FILE_NAME}.rs")).arg("-Zthreads=50").output(&bin_name);

            if is_windows_msvc() {
                rustc.arg("-Clink-arg=/Brepro");
            }

            rustc.run();

            let current = Rc::new(rfs::read(&bin_name));
            reference.get_or_insert(Rc::clone(&current));

            assert_eq!(Some(current), reference);
        });
    }
}
