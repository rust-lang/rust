//@ revisions: traditional next_solver
//@ [next_solver] compile-flags: -Znext-solver
//@ check-pass

use std::marker::PhantomData;

pub trait Receiver {
    type Target: ?Sized;
}

pub trait Deref: Receiver<Target = <Self as Deref>::Target> {
    type Target: ?Sized;
    fn deref(&self) -> &<Self as Deref>::Target;
}

impl<T: Deref> Receiver for T {
    type Target = <T as Deref>::Target;
}

// ===
pub struct Type<Id, T>(PhantomData<(Id, T)>);
pub struct AliasRef<Id, T: TypePtr<Id = Id>>(PhantomData<(Id, T)>);

pub trait TypePtr: Deref<Target = Type<<Self as TypePtr>::Id, Self>> + Sized {
    // ^ the impl head here provides the first candidate
    // <T as Deref>::Target := Type<<T as TypePtr>::Id>
    type Id;
}

pub struct Alias<Id, T>(PhantomData<(Id, T)>);

impl<Id, T> Deref for Alias<Id, T>
where
    T: TypePtr<Id = Id> + Deref<Target = Type<Id, T>>,
    // ^ the impl head here provides the second candidate
    // <T as Deref>::Target := Type<Id, T>
    // and additionally a normalisation is mandatory due to
    // the following supertrait relation trait
    // Deref: Receiver<Target = <Self as Deref>::Target>
{
    type Target = AliasRef<Id, T>;

    fn deref(&self) -> &<Self as Deref>::Target {
        todo!()
    }
}

fn main() {}
