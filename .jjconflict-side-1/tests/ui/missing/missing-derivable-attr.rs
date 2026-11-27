trait MyEq {
    fn eq(&self, other: &Self) -> bool;
}

struct A {
    x: isize
}

impl MyEq for isize {
    fn eq(&self, other: &isize) -> bool { *self == *other }
}

impl MyEq for A {}  //~ ERROR not all trait items implemented, missing: `eq`

fn main() {
}
