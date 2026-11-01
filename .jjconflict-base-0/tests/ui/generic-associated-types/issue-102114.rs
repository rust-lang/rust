//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait A {
    type B<'b>;
    fn a() -> Self::B<'static>;
}

struct C;

struct Wrapper<T>(T);

impl A for C {
    type B<T> = Wrapper<T>;
    //~^ ERROR type `B` has 1 type parameter but its trait declaration has 0 type parameters
    fn a() -> Self::B<'static> {}
}

fn main() {}
