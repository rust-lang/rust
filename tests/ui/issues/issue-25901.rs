struct A;
struct B;

static S: &'static B = &A;
//~^ ERROR cannot call conditionally-const method
//~| ERROR `Deref` is not yet stable as a const trait

use std::ops::Deref;

impl Deref for A {
    type Target = B;
    fn deref(&self)->&B { static B_: B = B; &B_ }
}

fn main(){}
