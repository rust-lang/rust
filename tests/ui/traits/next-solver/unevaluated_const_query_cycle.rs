//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/249

const CONST: &str = "hi";

trait ToUnit {
    type Assoc;
}
impl<T> ToUnit for T {
    type Assoc = ();
}

fn foo()
where
    <[u8; CONST.len()] as ToUnit>::Assoc: Sized,
{}

fn main(){}
