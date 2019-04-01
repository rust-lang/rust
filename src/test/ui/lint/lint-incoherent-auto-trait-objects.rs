// ignore-tidy-linelength

trait Foo {}

impl Foo for dyn Send {}

impl Foo for dyn Send + Send {}
//~^ ERROR conflicting implementations
//~| WARN this was previously accepted
//~| WARN hard error

impl Foo for dyn Send + Sync {}

impl Foo for dyn Sync + Send {}
//~^ ERROR conflicting implementations
//~| WARN this was previously accepted
//~| WARN hard error

impl Foo for dyn Send + Sync + Send {}
//~^ ERROR conflicting implementations
//~| WARN this was previously accepted
//~| WARN hard error

fn main() {}
