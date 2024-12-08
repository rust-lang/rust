//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// This goal is also possible w/ a GAT, but lazy_type_alias
// makes the behavior a bit more readable.
#![feature(lazy_type_alias)]
//~^ WARN the feature `lazy_type_alias` is incomplete

struct Wr<T>(T);
trait Foo {}
impl Foo for Wr<i32> {}

type Alias<T> = (T,)
    where Wr<T>: Foo;

fn hello<T>() where Alias<T>: Into<(T,)>, Wr<T>: Foo {}

fn main() {
    // When calling `hello`, proving `Alias<?0>: Into<(?0,)>` will require
    // normalizing the self type of the goal. This will emit the where
    // clause `Wr<?0>: Foo`, which constrains `?0` in both the self type
    // *and* the non-self part of the goal. That used to trigger a debug
    // assertion.
    hello::<_>();
}
