//@ check-pass

enum E {
    X(),
    Y,
}

fn main() {
    let _ = if let E::X() = E::X() { 1 } else { 2 };
}
