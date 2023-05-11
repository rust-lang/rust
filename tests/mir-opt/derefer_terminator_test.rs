// unit-test: Derefer
// EMIT_MIR derefer_terminator_test.main.Derefer.diff
// ignore-wasm32

fn main() {
    let b = foo();
    let d = foo();
    match ****(&&&&b) {
        true => {let x = 5;},
        false => {}
    }
    let y = 42;
}

fn foo() -> bool {
    true
}
