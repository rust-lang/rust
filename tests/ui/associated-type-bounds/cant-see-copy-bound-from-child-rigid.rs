//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Id {
    type This: ?Sized;
}

trait Trait {
    type Assoc: Id<This: Copy>;
}

// We can't see use the `T::Assoc::This: Copy` bound to prove `T::Assoc: Copy`
fn foo<T: Trait>(x: T::Assoc) -> (T::Assoc, T::Assoc)
where
    T::Assoc: Id<This = T::Assoc>,
{
    (x, x)
    //~^ ERROR use of moved value
}

fn main() {}
