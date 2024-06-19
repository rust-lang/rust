// This test checks the -Zremark-dir flag, which writes LLVM
// optimization remarks to the YAML format. When using PGO (Profile
// Guided Optimization), the Hotness attribute should be included in
// the output remark files.
// See https://github.com/rust-lang/rust/pull/114439

//@ needs-profiler-support
//@ ignore-cross-compile

use run_make_support::{run, llvm_profdata, rustc, invalid_utf8_contains};

fn main() {
    rustc().profile_generate("profdata").opt().input("foo.rs").output("foo").run();
    run("foo");
    llvm_profdata().merge().output("merged.profdata").input("profdata/default_15907418011457399462_0.profraw").run();
    rustc().profile_use("merged.profdata").opt().input("foo.rs").arg("-Cremark=all").arg("-Zremark-dir=profiles").run();
    // Check that PGO hotness is included in the remark files
    invalid_utf8_contains("profiles/foo.cba44757bc0621b9-cgu.0.opt.opt.yaml", "Hotness");
}
