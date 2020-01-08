#![feature(specialization)]
#![feature(optin_builtin_traits)]

trait MyTrait { }

impl<T> !MyTrait for T { }
impl MyTrait for u32 { } //~ ERROR conflicting implementations

fn main() { }
