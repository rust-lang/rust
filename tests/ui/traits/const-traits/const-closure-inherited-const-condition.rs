//@ check-pass
//@ revisions: next old
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(const_closures, const_trait_impl)]

const trait Foo {}

const fn qux<T: [const] Foo>() { (const || {})() }

fn main() {}
