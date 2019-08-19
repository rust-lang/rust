// run-pass
// Regression test for #36856.

// compile-flags:-g

fn g() -> bool {
    false
}

pub fn main() {
    let a = !g();
    if a != !g() {
        panic!();
    }
}
