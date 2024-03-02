pub trait Unsatisfied {}

#[repr(transparent)]
pub struct Bar<T: Unsatisfied>(T);

pub trait Foo {
    type Assoc;
}

extern "C" {
    pub fn lint_me() -> <() as Foo>::Assoc;
    //~^ ERROR trait `Foo` is not implemented for `()`

    pub fn lint_me_aswell() -> Bar<u32>;
    //~^ ERROR trait `Unsatisfied` is not implemented for `u32`
}

fn main() {}
