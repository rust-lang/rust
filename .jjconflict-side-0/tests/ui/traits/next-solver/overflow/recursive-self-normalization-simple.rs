//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// A variant of `recursive-self-normalization.rs` which passes if
// we treat inductive cycles as `NoSolution` in the trait solver.

trait Foo {
    type Assoc;
}

trait Bar {}
fn needs_bar<S: Bar>() {}

fn test<T: Foo<Assoc = <T as Foo>::Assoc>>() {
    //~^ ERROR overflow evaluating the requirement `<T as Foo>::Assoc == _`
}

fn main() {}
