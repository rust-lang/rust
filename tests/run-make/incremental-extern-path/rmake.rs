//@ only-linux
//@ needs-backends: llvm

use run_make_support::{rfs, rustc, rustc_minicore};

fn build(dir: &str, revision: &str) {
    rustc()
        .current_dir(dir)
        .input("foo.rs")
        .crate_name("repro")
        .crate_type("bin")
        .emit("obj")
        .cfg(revision)
        .extern_("minicore", "libminicore.rlib")
        .incremental("incremental")
        .debuginfo("2")
        .arg("-Cpanic=abort")
        .arg("-Zremap-cwd-prefix=.")
        .arg("-Zquery-dep-graph")
        .arg("-Zincremental-verify-ich")
        .run();
}

fn main() {
    rfs::create_dir("a");
    rustc_minicore().arg("-Cpanic=abort").output("a/libminicore.rlib").run();
    rfs::copy("foo.rs", "a/foo.rs");
    build("a", "cold");
    rfs::rename("a", "b");
    build("b", "reuse");
}
