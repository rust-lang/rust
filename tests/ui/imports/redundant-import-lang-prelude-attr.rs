// Check that we detect imports (of built-in attributes) that are redundant due to
// the language prelude and that we emit a reasonable diagnostic.
//~^^ NOTE the item `allow` is already defined by the extern prelude

// Note that we use the term "extern prelude" in the label even though "language prelude"
// would be more correct. However, it's not worth special-casing this.

// See also the discussion in <https://github.com/rust-lang/rust/pull/122954>.

//@ edition: 2018

#![deny(redundant_imports)]
//~^ NOTE the lint level is defined here

use allow; //~ ERROR the item `allow` is imported redundantly

#[allow(unused)]
fn main() {}
