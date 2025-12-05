// skip-filecheck
//@ test-mir-pass: Derefer
// EMIT_MIR derefer_terminator_test.main.Derefer.diff
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

fn main() {
    let b = foo();
    let d = foo();
    match ****(&&&&b) {
        true => {
            let x = 5;
        }
        false => {}
    }
    let y = 42;
}

fn foo() -> bool {
    true
}
