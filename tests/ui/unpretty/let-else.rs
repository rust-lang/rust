//@ compile-flags: -Zunpretty=expanded
//@ check-pass

macro_rules! expr {
    ($e:expr) => { $e };
}

fn main() {
    let _ = expr!(1 + 1) else { return; };
    let _ = expr!(loop {}) else { return; };
}
