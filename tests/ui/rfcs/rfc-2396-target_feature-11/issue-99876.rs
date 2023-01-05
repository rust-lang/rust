// check-pass

#![feature(target_feature_11)]

struct S<T>(T)
where
    [T; (|| {}, 1).1]: Copy;

fn main() {}
