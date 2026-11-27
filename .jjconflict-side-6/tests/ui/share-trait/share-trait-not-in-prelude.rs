#![feature(share_trait)]

#[derive(Clone)]
struct Alias;

impl Share for Alias {}
//~^ ERROR cannot find trait `Share` in this scope

fn main() {}
