//@error-in-other-file: the program aborted execution
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@compile-flags: -C panic=abort

fn main() {
    core::panic!("panicking from libcore");
}
