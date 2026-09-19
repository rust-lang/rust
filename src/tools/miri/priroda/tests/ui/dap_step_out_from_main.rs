//@ compile-flags: --dap

fn callee() {
    let inner = 1;
    let _ = inner;
}

fn main() {
    callee();
}
