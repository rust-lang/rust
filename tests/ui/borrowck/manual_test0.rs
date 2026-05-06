//fn foo<T, 'a, 'b, 'c, 'd>(val: T, _: &'b u64, _: &'a u32, _: &'c u32, _: &'d u32) where 'b : 'b, 'd : 'd {}

pub trait Moo {}
impl Moo for &u8 {}
pub struct Foo<Q: Moo>(Q);

fn test_handler<'a>(x: Foo<Moo<&'a u8>>) {}


fn main() {}