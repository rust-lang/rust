//@ check-pass

trait Family {
    type Member<'a>: for<'b> PartialEq<Self::Member<'b>>;
}

struct I32;

impl Family for I32 {
    type Member<'a> = i32;
}

fn main() {}
