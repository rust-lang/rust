//@ check-pass
//@ revisions: next old
//@[next] compile-flags: -Znext-solver

#![feature(const_closures, const_trait_impl)]

const trait Foo {}

const fn qux<T: [const] Foo>() { (const || {})() }

fn main() {}
