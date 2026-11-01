// Goals that no impl of a split family can satisfy, or that several can, are still reported.
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Tr {}
struct S<const N: usize>;

macro_rules! family {
    ($($n:literal)*) => { $(impl Tr for S<$n> {})* };
}
family!(0 1 2 3);

fn requires<T: Tr>() {}

fn without_bound<const N: usize>() {
    requires::<S<N>>(); //~ ERROR the trait bound `S<N>: Tr` is not satisfied
}

fn ambiguous() {
    requires::<S<_>>();
    //~^ ERROR type annotations needed
    //~| ERROR type annotations needed
}

fn main() {
    requires::<S<4>>(); //~ ERROR the trait bound `S<4>: Tr` is not satisfied
}
