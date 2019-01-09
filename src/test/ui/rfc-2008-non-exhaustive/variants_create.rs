#![feature(non_exhaustive)]

/*
 * The initial implementation of #[non_exhaustive] (RFC 2008) does not include support for
 * variants. See issue #44109 and PR 45394.
 */

pub enum NonExhaustiveVariants {
    #[non_exhaustive] Unit,
    //~^ ERROR #[non_exhaustive] is not yet supported on variants
    #[non_exhaustive] Tuple(u32),
    //~^ ERROR #[non_exhaustive] is not yet supported on variants
    #[non_exhaustive] Struct { field: u32 }
    //~^ ERROR #[non_exhaustive] is not yet supported on variants
}

fn main() { }
