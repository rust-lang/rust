// Regression test for #116464
// Checks that we do not suggest Trait<..., Assoc=arg> when the trait
// is referred to from one of its impls but do so at all other places

pub trait Trait<T> {
    type Assoc;
}

impl<T, S> Trait<T> for i32 {
    //~^ ERROR `S` is not constrained
    type Assoc = String;
}

// Should not trigger suggestion here...
impl<T, S> Trait<T, S> for () {}
//~^ ERROR trait takes 1 generic argument but 2 generic arguments were supplied

//... but should do so in all of the below cases except the last one
fn func<T: Trait<u32, String>>(t: T) -> impl Trait<(), i32> {
//~^ ERROR trait takes 1 generic argument but 2 generic arguments were supplied
//~| ERROR trait takes 1 generic argument but 2 generic arguments were supplied
//~| ERROR trait takes 1 generic argument but 2 generic arguments were supplied
    3
}

struct Struct<T: Trait<u32, String>> {
//~^ ERROR trait takes 1 generic argument but 2 generic arguments were supplied
    a: T
}

trait AnotherTrait<T: Trait<T, i32>> {}
//~^ ERROR trait takes 1 generic argument but 2 generic arguments were supplied

impl<T: Trait<u32, String>> Struct<T> {}
//~^ ERROR trait takes 1 generic argument but 2 generic arguments were supplied

// Test for self type. Should not trigger suggestion as it doesn't have an
// associated type
trait YetAnotherTrait {}
impl<T: Trait<u32, Assoc=String>, U> YetAnotherTrait for Struct<T, U> {}
//~^ ERROR struct takes 1 generic argument but 2 generic arguments were supplied


fn main() {
}
