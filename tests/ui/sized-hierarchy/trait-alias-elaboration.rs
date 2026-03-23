#![feature(sized_hierarchy, trait_alias)]
use std::marker::SizeOfVal;

// Trait aliases also have implicit `SizeOfVal` bounds, like traits. These are filtered out during
// elaboration of trait aliases when lowering `dyn TraitAlias` - however, if the user explicitly
// wrote `SizeOfVal` in the `dyn Trait` then that should still be an error so as not to accidentally
// accept this going forwards.

trait Qux = Clone;

type Foo = dyn Qux + SizeOfVal;
//~^ ERROR: only auto traits can be used as additional traits in a trait object

type Bar = dyn Qux;

fn main() {}
