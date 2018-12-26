// compile-pass
#![allow(unused)]

struct S;

impl S {
    fn early_and_type<'a, T>(self) -> &'a T { loop {} }
}

fn test() {
    S.early_and_type::<u16>();
}


fn main() {}
