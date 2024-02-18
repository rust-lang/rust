//@ check-pass
#![allow(unused_variables)]

struct Zeroes;
struct Foo<T>(T);

impl Into<[usize; 3]> for Zeroes {
    fn into(self) -> [usize; 3] {
        [0; 3]
    }
}

fn main() {
    let Foo([a, b, c]) = Foo(Zeroes.into());
}
