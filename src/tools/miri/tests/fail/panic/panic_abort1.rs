//@error-in-other-file: the program aborted execution
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\|.*::abort\(\).*" -> "| ABORT()"
//@compile-flags: -C panic=abort

fn main() {
    std::panic!("panicking from libstd");
}
