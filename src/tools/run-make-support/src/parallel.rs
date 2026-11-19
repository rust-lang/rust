use std::rc::Rc;

use crate::{bin_name, is_windows_msvc, rfs, run_in_tmpdir, rustc};

pub fn check_reproducibility(file: &str, args: &[&str]) {
    let mut reference = None;
    let bin_name = bin_name(file);

    for _ in 0..10 {
        // Tmp dir as previous runs affect output binary on windows.
        run_in_tmpdir(|| {
            let mut rustc = rustc();
            rustc.input(format!("{file}.rs")).output(&bin_name);

            for arg in args {
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
