//@ check-pass

type A = Box<dyn (Fn(u8) -> u8) + 'static + Send + Sync>; // OK (but see #39318)

fn main() {}
