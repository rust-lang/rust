//@error-in-other-file: the program aborted execution
//@compile-flags: -C panic=abort

fn main() {
    std::panic!("{}-panicking from libstd", 42);
}
