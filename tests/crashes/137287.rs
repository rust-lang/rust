//@ known-bug: #137287

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

fn main() {
    let x: Box<u32> = defining_scope::cast::<()>(Container { x: 0 }).x;
}
