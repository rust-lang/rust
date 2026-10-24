//@ check-pass

fn main() {
    for _ in [1, 2] {}
    let x = [1, 2];
    for _ in x {}
    for _ in [1.0, 2.0] {}
}
