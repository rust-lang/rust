// build-pass (FIXME(62277): could be check-pass?)

#![forbid(deprecated)]

#[deprecated = "oh no"]
#[derive(Default)]
struct X;

fn main() {}
