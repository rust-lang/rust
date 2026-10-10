//@ compile-flags: --dap

fn leaf() {
    let inside_leaf = 10;
    let _ = inside_leaf;
}

fn call_leaf() {
    leaf();
    let after_leaf = 20;
    let _ = after_leaf;
}

#[rustfmt::skip]
fn same_line() { fn callee() {} callee(); let after = 30; let _ = after; }

fn main() {
    call_leaf(); // DAP `next` starts here.
    same_line();
}
