// run-pass
trait Tr { type T; }
impl Tr for u8 { type T=(); }
struct S<I: Tr>(#[allow(unused_tuple_struct_fields)] I::T);

fn foo<I: Tr>(i: I::T) {
    S::<I>(i);
}

fn main() {
    foo::<u8>(());
}
