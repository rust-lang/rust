use std::ops::Generator;

fn gen() -> impl Generator<{}> {}
//~^ERROR constant provided when a type was expected
