//@ check-pass

struct S(
    #[rustfmt::skip] u8,
    u16,
    #[rustfmt::skip] u32,
);

fn main() {}
