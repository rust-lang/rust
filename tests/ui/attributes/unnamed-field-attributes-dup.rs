// Duplicate non-builtin attributes can be used on unnamed fields.

//@ check-pass

struct S (
    #[rustfmt::skip]
    #[rustfmt::skip]
    u8
);

fn main() {}
