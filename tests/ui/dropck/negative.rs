#![feature(negative_impls)]

struct NonDrop;
impl !Drop for NonDrop {}
//~^ ERROR negative `Drop` impls are not supported

fn main() {}
