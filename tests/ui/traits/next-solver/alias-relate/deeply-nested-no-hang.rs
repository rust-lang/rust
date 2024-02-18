//@ check-pass
//@ compile-flags: -Znext-solver
// regression test for trait-system-refactor-initiative#68
trait Identity {
    type Assoc: ?Sized;
}

impl<T: ?Sized> Identity for T {
    type Assoc = T;
}

type Id<T> = <T as Identity>::Assoc;

type Five<T> = Id<Id<Id<Id<Id<T>>>>>;
type Ty<T> = Five<Five<Five<Five<Five<T>>>>>;

trait Trait<T> {}

impl<T> Trait<T> for Ty<T> {}
impl Trait<u32> for Ty<i32> {}

fn main() {}
