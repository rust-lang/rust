// Regression test for #137287

mod defining_scope {
    use super::*;
    pub type Alias<T> = impl Sized;
    //~^ ERROR unconstrained opaque type
    //~| ERROR `impl Trait` in type aliases is unstable

    pub fn cast<T>(x: Container<Alias<T>, T>) -> Container<T, T> {
        x
        //~^ ERROR mismatched types
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
    //~^ ERROR conflicting implementations of trait `Trait<_>`
    type Assoc = usize;
}

fn main() {
    let x: Box<u32> = defining_scope::cast::<()>(Container { x: 0 }).x;
}
