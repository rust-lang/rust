use std::marker::CoerceShared; //~ ERROR use of unstable library feature `reborrow`
//~^ ERROR use of unstable library feature `reborrow`

#[derive(Clone, Copy)]
struct CustomRef<'a>(&'a ());

#[derive(std::marker::Reborrow, std::marker::CoerceShared)]
//~^ ERROR use of unstable library feature `reborrow`
//~| ERROR use of unstable library feature `reborrow`
#[coerce_shared(CustomRef<'a>)]
struct CustomMut<'a>(&'a mut ());

fn main() {}
