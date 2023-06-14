#![warn(clippy::missing_panics_doc)]

pub fn option_unwrap<T>(v: &[T]) -> &T {
    let o: Option<&T> = v.last();
    o.unwrap()
}

pub fn option_expect<T>(v: &[T]) -> &T {
    let o: Option<&T> = v.last();
    o.expect("passed an empty thing")
}

pub fn result_unwrap<T>(v: &[T]) -> &T {
    let res: Result<&T, &str> = v.last().ok_or("oh noes");
    res.unwrap()
}

pub fn result_expect<T>(v: &[T]) -> &T {
    let res: Result<&T, &str> = v.last().ok_or("oh noes");
    res.expect("passed an empty thing")
}

pub fn last_unwrap(v: &[u32]) -> u32 {
    *v.last().unwrap()
}

pub fn last_expect(v: &[u32]) -> u32 {
    *v.last().expect("passed an empty thing")
}

fn main() {}
