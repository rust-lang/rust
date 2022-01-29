struct A;
struct B;

static S: &'static B = &A;
//~^ ERROR calls in statics are limited to constant functions

use std::ops::Deref;

impl Deref for A {
    type Target = B;
    fn deref(&self)->&B { static B_: B = B; &B_ }
}

fn main(){}
