//@ check-pass

fn main() {
    let (0 | (1 | _)) = 0;
    if let 0 | (1 | 2) = 0 {}
    if let x @ 0 | x @ (1 | 2) = 0 {}
}
