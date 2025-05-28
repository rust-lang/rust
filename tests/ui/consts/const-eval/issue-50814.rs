//@ build-fail
//@ dont-require-annotations: NOTE

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
    //~^ ERROR attempt to compute `u8::MAX + u8::MAX`, which would overflow
    //~| ERROR attempt to compute `u8::MAX + u8::MAX`, which would overflow
}

fn foo<T>(_: T) -> &'static u8 {
    &Sum::<U8, U8>::MAX
    //~^ NOTE constant
}

fn main() {
    foo(0);
}
