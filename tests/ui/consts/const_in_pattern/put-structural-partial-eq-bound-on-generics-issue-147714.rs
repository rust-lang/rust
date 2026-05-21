// Regression test for https://github.com/rust-lang/rust/issues/147714.

#[allow(dead_code)]
#[derive(PartialEq)]
enum Thing<T> {
    A(T),
    B,
}

struct Incomparable;

impl PartialEq for Thing<Incomparable> {
    fn eq(&self, _: &Self) -> bool {
        panic!()
    }
}

const X: Thing<Incomparable> = Thing::B;

fn main() {
    if let X = X { //~ ERROR constant of non-structural type `Thing<Incomparable>` in a pattern
        println!("equal");
    }
}
