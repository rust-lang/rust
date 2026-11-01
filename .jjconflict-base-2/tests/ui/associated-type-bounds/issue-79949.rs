//@ check-pass

#![allow(incomplete_features)]
trait MP {
    type T<'a>;
}
struct S(String);
impl MP for S {
    type T<'a> = &'a str;
}

trait SR: MP {
    fn sr<IM>(&self) -> i32
    where
        for<'a> IM: T<T: U<<Self as MP>::T<'a>>>;
}

trait T {
    type T;
}
trait U<X> {}

fn main() {}
