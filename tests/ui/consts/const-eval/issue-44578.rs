//@ build-fail

trait Foo {
    const AMT: usize;
}

enum Bar<A, B> {
    First(A),
    Second(B),
}

impl<A: Foo, B: Foo> Foo for Bar<A, B> {
    const AMT: usize = [A::AMT][(A::AMT > B::AMT) as usize]; //~ERROR evaluation of `<Bar<u16, u8> as Foo>::AMT` failed
}

impl Foo for u8 {
    const AMT: usize = 1;
}

impl Foo for u16 {
    const AMT: usize = 2;
}

fn main() {
    println!("{}", <Bar<u16, u8> as Foo>::AMT);
    //~^ constant
}
