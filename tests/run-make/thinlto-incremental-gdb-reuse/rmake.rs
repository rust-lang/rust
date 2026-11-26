//! Changing one module should let ThinLTO reuse the other module's object file.
//! Both modules define the same GDB script global. If their order changes, LLVM
//! can pick a different copy of it and reoptimize the unchanged module too.
//! Here, we change each module in turn to exercise both input orders.

//@ needs-backends: llvm
//@ needs-debug-gdb-scripts

use std::collections::BTreeSet;
use std::path::PathBuf;

use run_make_support::{has_suffix, llvm_readobj, rfs, rustc, shallow_find_files};

fn main() {
    check_build("cold", &[], 2);
    llvm_readobj()
        .arg("--sections")
        .input("libindependent.rlib")
        .run()
        .assert_stdout_contains(".debug_gdb_scripts");
    check_build("changed_a", &["change_a"], 1);
    check_build("changed_b", &["change_a", "change_b"], 1);
    check_build("unchanged", &["change_a", "change_b"], 0);
}

fn check_build(revision: &str, cfgs: &[&str], expected_fresh: usize) {
    let mut compiler = rustc();
    compiler
        .input("independent.rs")
        .crate_type("rlib")
        .opt()
        .debuginfo("2")
        .codegen_units(2)
        .incremental("incremental")
        .arg("-Csave-temps");
    for cfg in cfgs {
        compiler.cfg(cfg);
    }
    compiler.run();

    let fresh = take_cgus(".no-opt.bc");
    assert_eq!(fresh.len(), expected_fresh, "fresh CGUs in {revision}: {fresh:?}");
    assert_eq!(take_cgus(".thin-lto-input.bc"), fresh, "CGUs sent to ThinLTO in {revision}");
}

// Strip the snapshot suffix so A.no-opt.bc and A.thin-lto-input.bc both identify A.
// All builds share an output directory. Delete each matching snapshot after
// recording its name, or a file from an earlier build could make a reused unit
// look as though it was optimized again.
fn take_cgus(suffix: &str) -> BTreeSet<PathBuf> {
    shallow_find_files(".", |p| has_suffix(p, suffix))
        .into_iter()
        .map(|p| {
            rfs::remove_file(&p);
            p.with_extension("").with_extension("")
        })
        .collect()
}
