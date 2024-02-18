//@ check-pass
// issue: 116794

// Uncaptured lifetimes should not be required to be live.

struct Invariant<T>(*mut T);

fn opaque<'a: 'a>(_: &'a str) -> Invariant<impl Sized> {
    Invariant(&mut ())
}

fn main() {
    let x = opaque(&String::new());
    drop(x);
}
