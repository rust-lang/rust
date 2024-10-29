//@ check-pass

const _: fn(&String) = |s| { &*s as &str; };

fn main() {}
