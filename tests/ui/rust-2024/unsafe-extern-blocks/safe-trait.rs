//@ revisions: gated ungated
#![cfg_attr(gated, feature(unsafe_extern_blocks))]

safe trait Foo {}
//~^ ERROR expected one of `!` or `::`, found keyword `trait`

fn main() {}
