// run-rustfix
#[derive(Debug, Default, Eq, PartialEq)]
struct A {
    b: u32,
    c: u64,
    d: usize,
}

fn main() {
    let q = A { c: 5 .. Default::default() };
    //~^ ERROR mismatched types
    //~| ERROR missing fields
    //~| HELP separate the last named field with a comma
    let r = A { c: 5, .. Default::default() };
    assert_eq!(q, r);
}
