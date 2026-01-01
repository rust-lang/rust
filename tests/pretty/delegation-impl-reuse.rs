//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:delegation-impl-reuse.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]

trait Trait<T> {
    fn foo(&self) {}
    fn bar(&self) {}
    fn baz(&self) {}
}

struct S;

reuse impl Trait<{ struct S; 0 }> for S { self.0 }

fn main() {}
