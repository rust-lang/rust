// run-pass
// compile-flags:-g
// ignore-asmjs wasm2js does not support source maps yet

fn helper<F: FnOnce(usize) -> bool>(_f: F) {
    print!("");
}

fn main() {
    let cond = 0;
    helper(|v| v == cond)
}
