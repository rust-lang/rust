//@error-in-other-file: the program aborted execution
//@compile-flags: -C panic=abort

fn main() {
    core::panic!("{}-panicking from libcore", 42);
}
