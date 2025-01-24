//@error-in-other-file: the program aborted execution
//@normalize-stderr-test: "unsafe \{ libc::abort\(\) \}|crate::intrinsics::abort\(\);" -> "ABORT();"
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""

#![feature(abort_unwind)]

fn main() {
    std::panic::abort_unwind(|| panic!("PANIC!!!"));
}
