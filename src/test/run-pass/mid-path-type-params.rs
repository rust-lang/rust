struct S<T> {
    contents: T,
}

impl<T> S<T> {
    fn new<U>(x: T, _: U) -> S<T> {
        S {
            contents: x,
        }
    }
}

trait Trait<T> {
    fn new<U>(x: T, y: U) -> Self;
}

struct S2 {
    contents: int,
}

impl Trait<int> for S2 {
    fn new<U>(x: int, _: U) -> S2 {
        S2 {
            contents: x,
        }
    }
}

fn main() {
    let _ = S::<int>::new::<float>(1, 1.0);
    let _: S2 = Trait::<int>::new::<float>(1, 1.0);
}

