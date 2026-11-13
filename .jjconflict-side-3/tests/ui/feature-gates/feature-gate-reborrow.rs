use std::marker::Reborrow; //~ ERROR use of unstable library feature `reborrow`
//~^ ERROR use of unstable library feature `reborrow`

#[derive(std::marker::Reborrow)] //~ ERROR use of unstable library feature `reborrow`
struct CustomMut<'a>(&'a mut ());

fn main() {}
