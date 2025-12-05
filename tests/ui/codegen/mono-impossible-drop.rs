//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ compile-flags: -Clink-dead-code=on --crate-type=lib
//@ build-pass

#![feature(trivial_bounds)]
#![allow(trivial_bounds)]

// Make sure we don't monomorphize the drop impl for `Baz`, since it has predicates
// that don't hold under a reveal-all param env.

trait Foo {
    type Assoc;
}

struct Bar;

struct Baz(<Bar as Foo>::Assoc)
where
    Bar: Foo;
