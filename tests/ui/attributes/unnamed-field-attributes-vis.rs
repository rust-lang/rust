// Unnamed fields don't lose their visibility due to non-builtin attributes on them.

//@ check-pass

mod m {
    pub struct S(#[rustfmt::skip] pub u8);
}

fn main() {
    m::S(0);
}
