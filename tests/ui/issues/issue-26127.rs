//@ run-pass
trait Tr { type T; }
impl Tr for u8 { type T=(); }
struct S<I: Tr>(#[allow(dead_code)] I::T);

fn foo<I: Tr>(i: I::T) {
    S::<I>(i);
}

fn main() {
    foo::<u8>(());
}
