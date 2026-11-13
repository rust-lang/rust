// Make sure we don't ICE if `layout_of` encounters an alias
// which is rigid due to a malformed program. A regression test
// for #152545.
//
// This specific ICE happens in the `KnownPanicsLint` visitor.

//@ compile-flags: --crate-type=rlib
trait Foo {
    type Assoc;
}

// The trait solver only treats missng associated items
// as rigid if the self-type is known to be unsized.
impl Foo for str {}
//~^ ERROR not all trait items implemented

fn foo(_: [u32; std::mem::size_of::<<str as Foo>::Assoc>()]) {}
