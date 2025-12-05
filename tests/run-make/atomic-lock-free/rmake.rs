// This tests ensure that atomic types are never lowered into runtime library calls that are not
// guaranteed to be lock-free.

//@ only-linux

use run_make_support::{llvm_components_contain, llvm_readobj, rustc};

fn compile(target: &str) {
    rustc().input("atomic_lock_free.rs").target(target).run();
}

fn check() {
    llvm_readobj()
        .symbols()
        .input("libatomic_lock_free.rlib")
        .run()
        .assert_stdout_not_contains("__atomic_fetch_add");
}

fn compile_and_check(target: &str) {
    compile(target);
    check();
}

fn main() {
    if llvm_components_contain("x86") {
        compile_and_check("i686-unknown-linux-gnu");
        compile_and_check("x86_64-unknown-linux-gnu");
    }
    if llvm_components_contain("arm") {
        compile_and_check("arm-unknown-linux-gnueabi");
        compile_and_check("arm-unknown-linux-gnueabihf");
        compile_and_check("armv7-unknown-linux-gnueabihf");
        compile_and_check("thumbv7neon-unknown-linux-gnueabihf");
    }
    if llvm_components_contain("aarch64") {
        compile_and_check("aarch64-unknown-linux-gnu");
    }
    if llvm_components_contain("mips") {
        compile_and_check("mips-unknown-linux-gnu");
        compile_and_check("mipsel-unknown-linux-gnu");
    }
    if llvm_components_contain("powerpc") {
        compile_and_check("powerpc-unknown-linux-gnu");
        compile_and_check("powerpc-unknown-linux-gnuspe");
        compile_and_check("powerpc64-unknown-linux-gnu");
        compile_and_check("powerpc64le-unknown-linux-gnu");
    }
    if llvm_components_contain("systemz") {
        compile_and_check("s390x-unknown-linux-gnu");
    }
}
