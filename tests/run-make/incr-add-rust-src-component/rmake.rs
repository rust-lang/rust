//! Regression test for rust-lang/rust#70924. Check that if we add the `rust-src` component in
//! between two incremental compiles, that the compiler doesn't ICE on the second invocation.
//!
//! This test uses symbolic links to save testing time.
//!
//! The way this test works is that, for every prefix in `root/lib/rustlib/src`, link all of prefix
//! parent content, then remove the prefix, then loop on the next prefix. This way, we basically
//! create a copy of the context around `root/lib/rustlib/src`, and can freely add/remove the src
//! component itself.

//@ ignore-cross-compile
// Reason: test needs to run.

//@ needs-symlink
// Reason: test needs symlink to create stub directories and files.

use std::path::Path;

use run_make_support::rfs::read_dir_entries;
use run_make_support::{bare_rustc, path, rfs, run};

#[derive(Debug, Copy, Clone)]
struct Symlink<'a, 'b> {
    src_dir: &'a Path,
    dst_dir: &'b Path,
}

fn shallow_symlink_dir<'a, 'b>(Symlink { src_dir, dst_dir }: Symlink<'a, 'b>) {
    eprintln!(
        "shallow_symlink_dir: src_dir={} -> dst_dir={}",
        src_dir.display(),
        dst_dir.display()
    );

    read_dir_entries(src_dir, |src_path| {
        let src_metadata = rfs::symlink_metadata(src_path);
        let filename = src_path.file_name().unwrap();
        if src_metadata.is_dir() {
            rfs::symlink_dir(src_path, dst_dir.join(filename));
        } else if src_metadata.is_file() {
            rfs::symlink_file(src_path, dst_dir.join(filename));
        } else if src_metadata.is_symlink() {
            rfs::copy_symlink(src_path, dst_dir.join(filename));
        }
    });
}

fn recreate_dir(path: &Path) {
    rfs::recursive_remove(path);
    rfs::create_dir(path);
}

fn main() {
    let sysroot = bare_rustc().print("sysroot").run().stdout_utf8();
    let sysroot = sysroot.trim();
    let sysroot = path(sysroot);

    let incr = path("incr");

    let fakeroot = path("fakeroot");
    rfs::create_dir(&fakeroot);

    shallow_symlink_dir(Symlink { src_dir: &sysroot, dst_dir: &fakeroot });
    recreate_dir(&fakeroot.join("lib"));

    shallow_symlink_dir(Symlink { src_dir: &sysroot.join("lib"), dst_dir: &fakeroot.join("lib") });
    recreate_dir(&fakeroot.join("lib").join("rustlib"));

    shallow_symlink_dir(Symlink {
        src_dir: &sysroot.join("lib").join("rustlib"),
        dst_dir: &fakeroot.join("lib").join("rustlib"),
    });
    recreate_dir(&fakeroot.join("lib").join("rustlib").join("src"));

    shallow_symlink_dir(Symlink {
        src_dir: &sysroot.join("lib").join("rustlib").join("src"),
        dst_dir: &fakeroot.join("lib").join("rustlib").join("src"),
    });

    rfs::recursive_remove(&fakeroot.join("lib").join("rustlib").join("src").join("rust"));

    let run_incr_rustc = || {
        bare_rustc()
            .sysroot(&fakeroot)
            .arg("-C")
            .arg(format!("incremental={}", incr.to_str().unwrap()))
            .input("main.rs")
            .run();
    };

    // Run rustc w/ incremental once...
    run_incr_rustc();

    // NOTE: the Makefile version of this used `$SYSROOT/lib/rustlib/src/rust/src/libstd/lib.rs`,
    // but that actually got moved around and reorganized over the years. As of Dec 2024, the
    // rust-src component is more like (specific for our purposes):
    //
    // ```
    // $SYSROOT/lib/rustlib/src/rust/
    //     library/std/src/lib.rs
    //     src/
    // ```
    rfs::create_dir_all(
        &fakeroot
            .join("lib")
            .join("rustlib")
            .join("src")
            .join("rust")
            .join("library")
            .join("std")
            .join("src"),
    );
    rfs::write(
        &fakeroot
            .join("lib")
            .join("rustlib")
            .join("src")
            .join("rust")
            .join("library")
            .join("std")
            .join("src")
            .join("lib.rs"),
        b"",
    );

    // ... and a second time.
    run_incr_rustc();

    // Basic sanity check that the compiled binary can run.
    run("main");
}
