//@ check-pass
// Check that we detect imports that are redundant due to the language prelude
// and that we emit a reasonable diagnostic.

// Note that we use the term "extern prelude" in the label even though "language prelude"
// would be more correct. However, it's not worth special-casing this.

// See also the discussion in <https://github.com/rust-lang/rust/pull/122954>.

#![deny(unused_imports)]

use std::primitive::u8;
//FIXME(unused_imports): ~^ ERROR the item `u8` is imported redundantly

const _: u8 = 0;

fn main() {}
