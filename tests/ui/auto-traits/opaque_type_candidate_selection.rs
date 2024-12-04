//@revisions: old next
//@[next] compile-flags: -Znext-solver

//! used to ICE: #119272

#![feature(type_alias_impl_trait)]
mod defining_scope {
    use super::*;
    pub type Alias<T> = impl Sized;

    pub fn cast<T>(x: Container<Alias<T>, T>) -> Container<T, T> {
        //[next]~^ ERROR: type annotations needed
        x
    }
}

struct Container<T: Trait<U>, U> {
    x: <T as Trait<U>>::Assoc,
}

trait Trait<T> {
    type Assoc;
}

impl<T> Trait<T> for T {
    type Assoc = Box<u32>;
}
impl<T> Trait<T> for defining_scope::Alias<T> {
    //~^ ERROR: conflicting implementations
    type Assoc = usize;
}

fn main() {}
