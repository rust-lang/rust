// Basic test for traits inheriting from the builtin kinds.

trait Foo : Send { }

impl <T: Sync+'static> Foo for T { }
//~^ ERROR `T` cannot be sent between threads safely

fn main() { }
