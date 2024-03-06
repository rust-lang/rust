//@ revisions: current next
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
