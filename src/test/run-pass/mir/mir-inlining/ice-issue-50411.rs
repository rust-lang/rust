// run-pass
// compile-flags:-Zmir-opt-level=3

fn main() {
    let _ = (0 .. 1).filter(|_| [1].iter().all(|_| true)).count();
}
