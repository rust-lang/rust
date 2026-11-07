// make sure that an *unused* broken const triggers an error even in a check build

//@ compile-flags: --emit=dep-info,metadata

const FOO: i32 = [][0];
//~^ ERROR index out of bounds

fn main() {}
