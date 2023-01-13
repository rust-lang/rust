struct A;
struct B;

static S: &'static B = &A;
//~^ ERROR the trait bound

use std::ops::Deref;

impl Deref for A {
    type Target = B;
    fn deref(&self)->&B { static B_: B = B; &B_ }
}

fn main(){}
