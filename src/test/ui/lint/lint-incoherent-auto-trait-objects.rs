// ignore-tidy-linelength

trait Foo {}

impl Foo for dyn Send {}

impl Foo for dyn Send + Send {}
//~^ ERROR conflicting implementations
//~| hard error
//~^^^ WARNING duplicate auto trait `Send` found in type parameter bounds [duplicate_auto_traits_in_bounds]

impl Foo for dyn Send + Sync {}

impl Foo for dyn Sync + Send {}
//~^ ERROR conflicting implementations
//~| hard error

impl Foo for dyn Send + Sync + Send {}
//~^ ERROR conflicting implementations
//~| hard error
//~^^^ WARNING duplicate auto trait `Send` found in type parameter bounds [duplicate_auto_traits_in_bounds]

fn main() {}
