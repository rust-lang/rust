// mir validation can't cope with `mem::uninitialized()`, so this test fails with validation & full-MIR.
// compile-flags: -Zmir-emit-validate=0

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
}
