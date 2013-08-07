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

fn main() {
    let _ = S::<int>::new::<float>(1, 1.0);
}

