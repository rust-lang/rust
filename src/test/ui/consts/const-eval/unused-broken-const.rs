// make sure that an *unused* broken const triggers an error even in a check build

// compile-flags: --emit=dep-info,metadata

const FOO: i32 = [][0];
//~^ ERROR any use of this value will cause an error
//~| WARN this was previously accepted by the compiler but is being phased out

fn main() {}
