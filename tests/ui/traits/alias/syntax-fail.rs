#![feature(trait_alias)]

trait Foo {}
#[rustc_auto_trait]
//~^ ERROR attribute should be applied to a trait
//~| ERROR `#[rustc_auto_trait]` is used to mark auto traits, only intended to be used in `core`
trait A = Foo;
unsafe trait B = Foo; //~ ERROR trait aliases cannot be `unsafe`

trait C: Ord = Eq; //~ ERROR bounds are not allowed on trait aliases
trait D: = Eq; //~ ERROR bounds are not allowed on trait aliases

fn main() {}
