// pretty-compare-only
// pretty-mode:hir
// pp-exact:never_patterns-hir.pp
#![feature(never_patterns)]
#![allow(incomplete_features)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;

enum Void { }

fn foo<'_>(res: &'_ Result<u32, Void>)
    -> &'_ u32 { match res { Ok(x) => x, Err(!), } }

fn main() { foo(&Ok(0)); }
