//@ check-pass

struct S<T>(T)
where
    [T; (|| {}, 1).1]: Copy;

fn main() {}
