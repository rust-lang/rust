// revisions:rpass1

#![feature(const_generics)]

struct Struct<T>(T);

impl<T, const N: usize> Struct<[T; N]> {
    fn f() {}
    fn g() { Self::f(); }
}

fn main() {
    Struct::<[u32; 3]>::g();
}
