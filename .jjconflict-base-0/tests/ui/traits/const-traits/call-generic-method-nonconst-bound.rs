//@ check-pass

struct S;

impl PartialEq for S {
    fn eq(&self, _: &S) -> bool {
        true
    }
}

const fn equals_self<T: PartialEq>(t: &T) -> bool {
    true
}

pub const EQ: bool = equals_self(&S);

fn main() {}
