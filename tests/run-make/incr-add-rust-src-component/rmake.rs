// rust-lang/rust#70924: Test that if we add rust-src component in between
// two incremental compiles, the compiler does not ICE on the second.
// Remove the rust-src part of the sysroot for the *first* build.
// Then put in a copy of the rust-src
// component for the second build, in order to expose the ICE from issue #70924.
// See https://github.com/rust-lang/rust/pull/72952

//@ needs-symlink

//FIXME(Oneirical): try on test-various

use run_make_support::{path, rfs, rustc};

fn main() {
    let sysroot = rustc().print("sysroot").run().stdout_utf8();
    let sysroot = sysroot.trim();
    rfs::create_dir("fakeroot");
    symlink_all_entries(&sysroot, "fakeroot");
    rfs::remove_file("fakeroot/lib");
    rfs::create_dir("fakeroot/lib");
    symlink_all_entries(path(&sysroot).join("lib"), "fakeroot/lib");
    rfs::remove_file("fakeroot/lib/rustlib");
    rfs::create_dir("fakeroot/lib/rustlib");
    symlink_all_entries(path(&sysroot).join("lib/rustlib"), "fakeroot/lib/rustlib");
    rfs::remove_file("fakeroot/lib/rustlib/src");
    rfs::create_dir("fakeroot/lib/rustlib/src");
    symlink_all_entries(path(&sysroot).join("lib/rustlib/src"), "fakeroot/lib/rustlib/src");
    rfs::remove_file("fakeroot/lib/rustlib/src/rust");
    rustc().sysroot("fakeroot").incremental("incr").input("main.rs").run();
    rfs::create_dir_all("fakeroot/lib/rustlib/src/rust/src/libstd");
    rfs::create_file("fakeroot/lib/rustlib/src/rust/src/libstd/lib.rs");
    rustc().sysroot("fakeroot").incremental("incr").input("main.rs").run();
}

fn symlink_all_entries<P: AsRef<std::path::Path>>(dir: P, fakepath: &str) {
    for found_path in rfs::shallow_find_dir_entries(dir) {
        rfs::create_symlink(&found_path, path(fakepath).join(found_path.file_name().unwrap()));
    }
}
