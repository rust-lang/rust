// rust-lang/rust#70924: Test that if we add rust-src component in between two
// incremental compiles, the compiler does not ICE on the second.
// Remove the rust-src part of the sysroot for the *first* build.
// Then put in a facsimile of the rust-src
// component for the second build, in order to expose the ICE from issue #70924.
// See https://github.com/rust-lang/rust/pull/72952

//FIXME(Oneirical): try on test-various and windows
//FIXME(Oneirical): check that the direct edit of the sysroot is not messing things up

use run_make_support::{path, rfs, rustc};

fn main() {
    let sysroot = rustc().print("sysroot").run().stdout_utf8();
    let sysroot = sysroot.trim();
    let sysroot = format!("{sysroot}-sysroot");
    rfs::remove_dir_all(path(&sysroot).join("lib/rustlib/src/rust"));
    rustc().arg("--sysroot").arg(&sysroot).incremental("incr").input("main.rs").run();
    rfs::create_dir_all(path(&sysroot).join("lib/rustlib/src/rust/src/libstd"));
    rfs::create_file(path(&sysroot).join("lib/rustlib/src/rust/src/libstd/lib.rs"));
    rustc().arg("--sysroot").arg(&sysroot).incremental("incr").input("main.rs").run();
}
