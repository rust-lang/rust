//@ needs-target-std
//
// Specifying how rustc outputs a file can be done in different ways, such as
// the output flag or the KIND=NAME syntax. However, some of these methods used
// to result in different hashes on output files even though they yielded the
// exact same result otherwise. This was fixed in #86045, and this test checks
// that the hash is only modified when the output is made different, such as by
// adding a new output type (in this test, metadata).
// See https://github.com/rust-lang/rust/issues/86044

use run_make_support::{diff, rfs, rustc};

fn main() {
    rfs::create_dir("emit");
    rfs::create_dir("emit/a");
    rfs::create_dir("emit/b");
    rfs::create_dir("emit/c");
    // The default output name.
    rustc().emit("link").input("foo.rs").run();
    // The output is named with the output flag.
    rustc().emit("link").output("emit/a/libfoo.rlib").input("foo.rs").run();
    // The output is named with link=NAME.
    rustc().emit("link=emit/b/libfoo.rlib").input("foo.rs").run();
    // The output is named with link=NAME, with an additional kind tacked on.
    rustc().emit("link=emit/c/libfoo.rlib,metadata").input("foo.rs").run();

    let base = rustc().arg("-Zls=root").input("libfoo.rlib").run().stdout_utf8();
    let a = rustc().arg("-Zls=root").input("emit/a/libfoo.rlib").run().stdout_utf8();
    let b = rustc().arg("-Zls=root").input("emit/b/libfoo.rlib").run().stdout_utf8();
    let c = rustc().arg("-Zls=root").input("emit/c/libfoo.rlib").run().stdout_utf8();
    // Both the output flag and link=NAME methods do not modify the hash of the output file.
    diff().expected_text("base", &base).actual_text("a", a).run();
    diff().expected_text("base", &base).actual_text("b", b).run();
    // However, having multiple types of outputs does modify the hash.
    diff().expected_text("base", &base).actual_text("c", c).run_fail();
}
