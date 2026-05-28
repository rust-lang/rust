//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for the fourth variant of trait-system-refactor-initiative#191.
// We previously didn't normalize `<() as Trait<T>>::Assoc<'a>` before generalizing
// here, resulting in an error.

trait Trait<T> {
    type Assoc<'a>;
}

impl<T> Trait<T> for () {
    type Assoc<'a> = &'a ();
}

fn foo<T>(x: Option<*mut T>) -> for<'a> fn(<() as Trait<T>>::Assoc<'a>) {
    |_| ()
}

fn main() {
    let mut x = None;
    let mut y = foo(x);
    x = Some(&mut y);
}
