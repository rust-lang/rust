#![feature(trait_alias)] // Enabled to reduce stderr output, but can be triggered even if disabled.
trait Trait {}
trait WithType {
    type Ctx;
}
trait Alias<T> = where T: Trait;

impl<T> WithType for T {
    type Ctx = dyn Alias<T>;
    //~^ ERROR at least one trait is required for an object type [E0224]
}
fn main() {}
