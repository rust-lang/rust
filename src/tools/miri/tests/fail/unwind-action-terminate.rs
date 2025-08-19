//@error-in-other-file: aborted execution
//@normalize-stderr-test: "\|.*::abort\(\).*" -> "| ABORT()"
//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""
extern "C" fn panic_abort() {
    panic!()
}

fn main() {
    panic_abort();
}
