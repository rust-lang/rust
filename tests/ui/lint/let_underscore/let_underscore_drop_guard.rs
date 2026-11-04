use std::mem::DropGuard;

struct Struct<T> {
    a: T,
}

fn main() {
    let _ = DropGuard::new((), |_| {});
    //~^ERROR non-binding let on a `DropGuard`
    _ = DropGuard::new((), |_| {});
    //~^ERROR non-binding let on a `DropGuard`

    let (_, _) = (DropGuard::new((), |_| {}), 1);
    //~^ERROR non-binding let on a `DropGuard`
    (_, _) = (DropGuard::new((), |_| {}), 1);
    //~^ERROR non-binding let on a `DropGuard`

    let (_a, Struct { a: _ }) = (0, Struct { a: DropGuard::new((), |_| {}) });
    //~^ERROR non-binding let on a `DropGuard`
    let _b;
    (_b, Struct { a: _ }) = (0, Struct { a: DropGuard::new((), |_| {}) });
    //~^ERROR non-binding let on a `DropGuard`
}
