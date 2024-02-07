// pp-exact
#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn foo(res: &Result<u32, Void>) -> &u32 { match res { Ok(x) => x, Err(!), } }

fn main() { foo(&Ok(0)); }
