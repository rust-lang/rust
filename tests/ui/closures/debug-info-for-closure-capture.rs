//@ run-pass
//@ compile-flags:-g

fn helper<F: FnOnce(usize) -> bool>(_f: F) {
    print!("");
}

fn main() {
    let cond = 0;
    helper(|v| v == cond)
}
