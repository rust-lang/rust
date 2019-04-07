// run-pass

enum Nat {
    S(Box<Nat>),
    Z
}
fn test(x: &mut Nat) {
    let mut p = &mut *x;
    loop {
        match p {
            &mut Nat::Z => break,
            &mut Nat::S(ref mut n) => p = &mut *n
        }
    }
}

fn main() {}
