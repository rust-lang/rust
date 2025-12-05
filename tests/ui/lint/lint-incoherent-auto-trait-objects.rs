trait Foo {}

impl Foo for dyn Send {}

impl Foo for dyn Send + Send {}
//~^ ERROR conflicting implementations

impl Foo for dyn Send + Sync {}

impl Foo for dyn Sync + Send {}
//~^ ERROR conflicting implementations

impl Foo for dyn Send + Sync + Send {}
//~^ ERROR conflicting implementations

fn main() {}
