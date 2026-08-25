//@ ignore-windows-gnu
// GNU Linker for Windows is non-deterministic. (from `reproducible-build-2` test in this suite)

use std::rc::Rc;

use run_make_support::{bin_name, is_windows_msvc, rfs, run_in_tmpdir, rustc};

/// Test that parallel compiler produces identical artifacts (binaries, metadata).
fn main() {
    const TESTS: &[(&str, &[&str])] = &[
        ("static-muts-issue-140413", &["-Zthreads=50"]),
        ("derives-issue-129094", &["-Zthreads=16", "-Copt-level=3"]),
        ("mir-alloc-ids-issue-154278", &["-Zthreads=60", "--emit=mir", "--crate-type=lib"]),
    ];

    for (file, args) in TESTS {
        let mut reference = None;
        let bin_name = bin_name(file);

        for _ in 0..100 {
            // Tmp dir as previous runs affect output binary on windows.
            run_in_tmpdir(|| {
                let mut rustc = rustc();
                rustc.input(format!("{file}.rs")).output(&bin_name);

                for arg in *args {
                    rustc.arg(arg);
                }

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
}
