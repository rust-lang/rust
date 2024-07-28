//! used to ICE: #119272

//@ check-pass

#![feature(type_alias_impl_trait)]
mod defining_scope {
    use super::*;
    pub type Alias<T> = impl Sized;

    pub fn cast<T>(x: Container<Alias<T>, T>) -> Container<T, T> {
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
    type Assoc = usize;
}

fn main() {}
