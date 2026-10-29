//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/141322>.

trait Trait<'a> {
    type Assoc;
}

struct Thing;

impl<'a> Trait<'a> for Thing {
    type Assoc = &'a i32;
}

fn wrap<T, U: for<'a> Trait<'a, Assoc = T>>() {}

fn foo() {
    wrap::<_, Thing>();
    //[next]~^ ERROR type mismatch resolving `<Thing as Trait<'a>>::Assoc == &i32
    //[current]~^^ ERROR mismatched types
}

fn main() {}
