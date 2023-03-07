// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![recursion_limit = "128"]
// this used to cause exponential code-size blowup during LLVM passes.

#![feature(test)]

extern crate test;

struct MayUnwind;

impl Drop for MayUnwind {
    fn drop(&mut self) {
        if test::black_box(false) {
            panic!()
        }
    }
}

struct DS<U> {
    may_unwind: MayUnwind,
    name: String,
    next: U,
}

fn add<U>(ds: DS<U>, name: String) -> DS<DS<U>> {
    DS {
        may_unwind: MayUnwind,
        name: "?".to_owned(),
        next: ds,
    }
}

fn main() {
    let deserializers = DS { may_unwind: MayUnwind, name: "?".to_owned(), next: () };
    let deserializers = add(deserializers, "?".to_owned());
    let deserializers = add(deserializers, "?".to_owned());
    let deserializers = add(deserializers, "?".to_owned());
    let deserializers = add(deserializers, "?".to_owned());
    let deserializers = add(deserializers, "?".to_owned());
    let deserializers = add(deserializers, "?".to_owned());
    let deserializers = add(deserializers, "?".to_owned()); // 0.7s
    let deserializers = add(deserializers, "?".to_owned()); // 1.3s
    let deserializers = add(deserializers, "?".to_owned()); // 2.4s
    let deserializers = add(deserializers, "?".to_owned()); // 6.7s
    let deserializers = add(deserializers, "?".to_owned()); // 26.0s
    let deserializers = add(deserializers, "?".to_owned()); // 114.0s
    let deserializers = add(deserializers, "?".to_owned()); // 228.0s
    let deserializers = add(deserializers, "?".to_owned()); // 400.0s
    let deserializers = add(deserializers, "?".to_owned()); // 800.0s
    let deserializers = add(deserializers, "?".to_owned()); // 1600.0s
    let deserializers = add(deserializers, "?".to_owned()); // 3200.0s
}
