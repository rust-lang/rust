// compile-flags: -C overflow-checks=on -O
// run-pass

fn main() {
    let _ = -(-0.0);
}
