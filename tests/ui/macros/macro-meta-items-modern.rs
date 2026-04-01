//@ check-pass

macro_rules! check { ($meta:meta) => () }

check!(meta(a b c d));
check!(meta[a b c d]);
check!(meta { a b c d });
check!(meta);
check!(meta = 0);

fn main() {}
