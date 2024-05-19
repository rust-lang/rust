trait Foo {}

impl Foo for dyn Send {}

impl Foo for dyn Send + Send {}
//~^ ERROR conflicting implementations
//~| hard error

impl Foo for dyn Send + Sync {}

impl Foo for dyn Sync + Send {}
//~^ ERROR conflicting implementations
//~| hard error

impl Foo for dyn Send + Sync + Send {}
//~^ ERROR conflicting implementations
//~| hard error

fn main() {}
