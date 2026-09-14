//@ compile-flags: --dap

fn callee() {
    let inner = 1;
    let _ = inner;
}

fn main() {
    callee();
    let after = 2;
    let _ = after;
}
