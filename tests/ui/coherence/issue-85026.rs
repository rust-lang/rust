#![feature(auto_traits)]
auto trait AutoTrait {}

// You cannot impl your own `dyn AutoTrait`.
impl dyn AutoTrait {} //~ERROR E0785

// You cannot impl someone else's `dyn AutoTrait`
impl dyn Unpin {} //~ERROR E0785

fn main() {}
