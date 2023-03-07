// run-pass
// Regression test for #36856.

// compile-flags:-g
// ignore-asmjs wasm2js does not support source maps yet

fn g() -> bool {
    false
}

pub fn main() {
    let a = !g();
    if a != !g() {
        panic!();
    }
}
