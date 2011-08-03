// -*- rust -*-
// error-pattern: Dynamically sized arguments must be passed by alias

mod foo {
    fn bar[T](f: T) -> int { ret 17; }
    type bar[U, T] = {a: int, b: U, c: T};
}

fn main() { }