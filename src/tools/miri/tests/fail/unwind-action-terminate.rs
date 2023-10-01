//@error-in-other-file: aborted execution
//@normalize-stderr-test: "unsafe \{ libc::abort\(\) \}|crate::intrinsics::abort\(\);" -> "ABORT();"
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""
#![feature(c_unwind)]

extern "C" fn panic_abort() {
    panic!()
}

fn main() {
    panic_abort();
}
