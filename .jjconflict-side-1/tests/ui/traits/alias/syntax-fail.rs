#![feature(trait_alias)]

trait Foo {}
auto trait A = Foo; //~ ERROR trait aliases cannot be `auto`
unsafe trait B = Foo; //~ ERROR trait aliases cannot be `unsafe`

trait C: Ord = Eq; //~ ERROR bounds are not allowed on trait aliases
trait D: = Eq; //~ ERROR bounds are not allowed on trait aliases

fn main() {}
