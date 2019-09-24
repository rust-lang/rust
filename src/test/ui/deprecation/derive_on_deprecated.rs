// build-pass (FIXME(62277): could be check-pass?)

#![deny(deprecated)]

#[deprecated = "oh no"]
#[derive(Default)]
struct X;

fn main() {}
