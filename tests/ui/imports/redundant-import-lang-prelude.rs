// Check that we detect imports that are redundant due to the language prelude
// and that we emit a reasonable diagnostic.
//~^^ NOTE the item `u8` is already defined by the extern prelude

// Note that we use the term "extern prelude" in the label even though "language prelude"
// would be more correct. However, it's not worth special-casing this.

// See also the discussion in <https://github.com/rust-lang/rust/pull/122954>.

#![deny(redundant_imports)]
//~^ NOTE the lint level is defined here

use std::primitive::u8;
//~^ ERROR the item `u8` is imported redundantly

const _: u8 = 0;

fn main() {}
