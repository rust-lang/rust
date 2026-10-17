pub trait Unsatisfied {}

#[repr(transparent)]
pub struct Bar<T: Unsatisfied>(T);

pub trait Foo {
    type Assoc;
}

extern "C" {
    pub fn lint_me() -> <() as Foo>::Assoc;
    //~^ ERROR: the trait bound `(): Foo` is not satisfied [E0277]

    pub fn lint_me_aswell() -> Bar<u32>;
    //~^ ERROR: the trait bound `u32: Unsatisfied` is not satisfied [E0277]
}

fn main() {}
