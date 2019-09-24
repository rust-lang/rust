// build-pass (FIXME(62277): could be check-pass?)

#![allow(bare_trait_objects)]

type A = Box<(Fn(u8) -> u8) + 'static + Send + Sync>; // OK (but see #39318)

fn main() {}
