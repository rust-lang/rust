// ignore-tidy-linelength

trait Foo {}

impl Foo for dyn Send {}

impl Foo for dyn Send + Send {}
//~^ ERROR conflicting implementations
//~| hard error
//~^^^ WARNING duplicate auto trait `std::marker::Send` found in trait object [duplicate_auto_traits_in_trait_objects]

impl Foo for dyn Send + Sync {}

impl Foo for dyn Sync + Send {}
//~^ ERROR conflicting implementations
//~| hard error

impl Foo for dyn Send + Sync + Send {}
//~^ ERROR conflicting implementations
//~| hard error
//~^^^ WARNING duplicate auto trait `std::marker::Send` found in trait object [duplicate_auto_traits_in_trait_objects]

fn main() {}
