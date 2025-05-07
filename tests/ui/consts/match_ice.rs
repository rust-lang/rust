// https://github.com/rust-lang/rust/issues/53708

struct S;

#[derive(PartialEq, Eq)]
struct T;

fn main() {
    const C: &S = &S;
    match C {
        C => {} //~ ERROR constant of non-structural type `S` in a pattern
    }
    const K: &T = &T;
    match K {
        K => {}
    }
}
