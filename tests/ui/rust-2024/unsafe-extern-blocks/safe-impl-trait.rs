//@ revisions: gated ungated
#![cfg_attr(gated, feature(unsafe_extern_blocks))]

trait Bar {}
safe impl Bar for () { }
//~^ ERROR expected one of `!` or `::`, found keyword `impl`

fn main() {}
