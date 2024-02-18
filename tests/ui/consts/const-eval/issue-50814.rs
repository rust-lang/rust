//@ build-fail

trait Unsigned {
    const MAX: u8;
}

struct U8(u8);
impl Unsigned for U8 {
    const MAX: u8 = 0xff;
}

struct Sum<A, B>(A, B);

impl<A: Unsigned, B: Unsigned> Unsigned for Sum<A, B> {
    const MAX: u8 = A::MAX + B::MAX;
    //~^ ERROR evaluation of `<Sum<U8, U8> as Unsigned>::MAX` failed
    //~| ERROR evaluation of `<Sum<U8, U8> as Unsigned>::MAX` failed
}

fn foo<T>(_: T) -> &'static u8 {
    &Sum::<U8, U8>::MAX
    //~^ constant
}

fn main() {
    foo(0);
}
