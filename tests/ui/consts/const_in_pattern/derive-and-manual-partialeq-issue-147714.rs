// `derive(PartialEq)`, which also implements `StructuralPartialEq`, must not allow the latter impl
// to be used with other non-derived implementations of `PartialEq`.
//
// Test case by theemathas from <https://github.com/rust-lang/rust/issues/147714>.

#[allow(dead_code)]
#[derive(PartialEq)]
enum Thing<T> {
    A(T),
    B,
}

struct Incomparable;

// This impl does not obey StructuralPartialEq's rules.
impl PartialEq for Thing<Incomparable> {
    fn eq(&self, _: &Self) -> bool {
        panic!()
    }
}

// This constant does not obey StructuralPartialEq's rules, so it should not
// implement StructuralPartialEq.
const X: Thing<Incomparable> = Thing::B;

fn main() {
    if let X = X {
        //~^ ERROR constant of non-structural type `Thing<Incomparable>` in a pattern
        println!("equal");
    }
}
