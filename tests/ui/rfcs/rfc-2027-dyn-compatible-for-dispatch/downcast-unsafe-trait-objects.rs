// Check that we if we get ahold of a dyn-incompatible trait
// object with auto traits and lifetimes, we can downcast it
//
//@ check-pass

#![feature(dyn_compatible_for_dispatch)]

trait Trait: Sized {}

fn downcast_auto(t: &(dyn Trait + Send)) -> &dyn Trait {
    t
}

fn downcast_lifetime<'a, 'b, 't>(t: &'a (dyn Trait + 't))
                                 -> &'b (dyn Trait + 't)
where
    'a: 'b,
    't: 'a + 'b,
{
    t
}

fn main() {}
