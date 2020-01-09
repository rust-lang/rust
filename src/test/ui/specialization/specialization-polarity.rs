// Make sure specialization cannot change impl polarity

#![feature(optin_builtin_traits)]
#![feature(negative_impls)]
#![feature(specialization)]

auto trait Foo {}

impl<T> Foo for T {}
impl !Foo for u8 {} //~ ERROR E0748

auto trait Bar {}

impl<T> !Bar for T {}
impl Bar for u8 {} //~ ERROR E0748

fn main() {}
