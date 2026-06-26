#![feature(sized_hierarchy, trait_alias)]
use std::marker::MetaSized;

// Trait aliases also have implicit `MetaSized` bounds, like traits. These are filtered out during
// elaboration of trait aliases when lowering `dyn TraitAlias` - however, if the user explicitly
// wrote `MetaSized` in the `dyn Trait` then that should still be an error so as not to accidentally
// accept this going forwards.

trait Qux = Clone;

type Foo = dyn Qux + MetaSized;
//~^ ERROR: only auto traits can be used as additional traits in a trait object

type Bar = dyn Qux;

fn main() {}
