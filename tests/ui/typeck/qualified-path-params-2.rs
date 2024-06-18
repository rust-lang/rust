// FIXME(fmease): Write a new description or remove test entirely.

struct S;

trait Tr {
    type A;
}

impl Tr for S {
    type A = S;
}

impl S {
    fn f<T>() {}
}

type A = <S as Tr>::A::f<u8>; //~ ERROR associated type `f` not found for `<S as Tr>::A`

fn main() {}
