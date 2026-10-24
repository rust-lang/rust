// Checks what happens when formatting the panic message panics.

//@ run-crash
//@ exec-env:RUST_BACKTRACE=0
//@ check-run-results
//@ error-pattern: panicked while processing panic
//@ normalize-stderr: "\n +[0-9]+:[^\n]+" -> ""
//@ normalize-stderr: "\n +at [^\n]+" -> ""
//@ normalize-stderr: "(core/src/panicking\.rs):[0-9]+:[0-9]+" -> "$1:$$LINE:$$COL"
//@ ignore-emscripten "RuntimeError" junk in output

use std::fmt::{Display, self};

struct MyStruct;

impl Display for MyStruct {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

fn main() {
    let instance = MyStruct;
    panic!("this is wrong: {}", instance);
}
