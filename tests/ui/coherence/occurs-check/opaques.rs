//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// A regression test for #105787
#![feature(type_alias_impl_trait)]

pub type Alias<T> = impl Sized;

#[define_opaque(Alias)]
pub fn cast<T>(x: Container<Alias<T>, T>) -> Container<T, T> {
    //[next]~^ ERROR type annotations needed
    x
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
impl<T> Trait<T> for Alias<T> {
    //~^ ERROR conflicting implementations of trait
    type Assoc = usize;
}

fn main() {
    let x: Box<u32> = cast::<()>(Container { x: 0 }).x;
    println!("{}", *x);
}
