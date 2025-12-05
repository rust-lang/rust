struct A;
struct B;

static S: &'static B = &A;
//~^ ERROR cannot perform non-const deref coercion

use std::ops::Deref;

impl Deref for A {
    type Target = B;
    fn deref(&self)->&B { static B_: B = B; &B_ }
}

fn main(){}
