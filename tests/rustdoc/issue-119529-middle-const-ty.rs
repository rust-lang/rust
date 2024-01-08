// This is a regression test for <https://github.com/rust-lang/rust/issues/119529>.

#![crate_name = "foo"]

use std::cmp::PartialEq;

// @has 'foo/type.Dyn.html'
// @has - '//*[@class="rust item-decl"]' 'pub type Dyn<Rhs> = dyn PartialEq<Rhs>;'
pub type Dyn<Rhs> = dyn PartialEq<Rhs>;
