// FIXME: Validation disabled due to https://github.com/rust-lang/rust/issues/54957
// compile-flags: -Zmiri-disable-validation

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Foo {
    A(&'static str),
    _B,
    _C,
}

pub fn main() {
    let mut b = std::collections::BTreeSet::new();
    b.insert(Foo::A("\'"));
    b.insert(Foo::A("/="));
    b.insert(Foo::A("#"));
    b.insert(Foo::A("0o"));
    assert!(b.remove(&Foo::A("/=")));
    assert!(!b.remove(&Foo::A("/=")));
}
