//@ needs-llvm-components: arm
//@ needs-rust-lld
use run_make_support::{
    llvm_filecheck, llvm_objdump, path, rfs, run, rustc, rustc_minicore, source_root,
};

// Test a thumb target calling arm functions. Doing so requires switching from thumb mode to arm
// mode, calling the arm code, then switching back to thumb mode. Depending on the thumb version,
// this happens using a special calling instruction, or by calling a generated thunk that performs
// the mode switching.
//
// In particular this tests that naked functions behave like normal functions. Before LLVM 22, a
// bug in LLVM caused thumb mode to be used unconditonally when symbols were .hidden, miscompiling
// calls to arm functions.
//
// - https://github.com/llvm/llvm-project/pull/181156
// - https://github.com/rust-lang/rust/issues/151946

fn main() {
    // Thumb calling thumb and arm.
    helper("thumbv5te", "thumbv5te-none-eabi");
    helper("thumbv4t", "thumbv4t-none-eabi");

    // Arm calling thumb and arm.
    helper("armv5te", "armv5te-none-eabi");
    helper("armv4t", "armv4t-none-eabi");
}

fn helper(prefix: &str, target: &str) {
    rustc_minicore().target(target).output("libminicore.rlib").run();

    rustc()
        .input("main.rs")
        .panic("abort")
        .link_arg("-Tlink.ld")
        .extern_("minicore", path("libminicore.rlib"))
        .target(target)
        .output(prefix)
        .run();

    let dump = llvm_objdump().disassemble().demangle().input(path(prefix)).run();

    eprintln!("{}", str::from_utf8(&dump.stdout()).unwrap());

    llvm_filecheck().patterns("main.rs").check_prefix(prefix).stdin_buf(dump.stdout()).run();
}
