#![feature(negative_impls)]
#![feature(min_specialization)]

use std::marker::Forget;
//~^ ERROR use of unstable library feature `forget_trait`
use std::mem;

struct A;
struct B;

impl !Forget for B {}
//~^ ERROR use of unstable library feature `forget_trait`

fn main() {
    mem::forget(A);
    mem::forget(B);
}
