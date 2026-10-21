//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait HasLifetime {
    type AtLifetime<'a>;
}

pub struct ExistentialLifetime<S: HasLifetime>(S::AtLifetime<'static>);

impl<S: HasLifetime> ExistentialLifetime<S> {
    fn new() -> ExistentialLifetime<S> {
        ExistentialLifetime(ExistentialLifetime(()))
        //~^ ERROR: type mismatch resolving `<S as HasLifetime>::AtLifetime<'static> == ExistentialLifetime<_>` [E0271]
    }
}

fn main() {}
