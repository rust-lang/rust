//@ compile-flags: -Zdeduplicate-diagnostics=yes
extern crate rustc_data_structures;
//~^ use of unstable library feature 'rustc_private'
//~| NOTE: issue #27812 <https://github.com/rust-lang/rust/issues/27812> for more information
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
extern crate rustc_macros;
//~^ use of unstable library feature 'rustc_private'
//~| NOTE: see issue #27812 <https://github.com/rust-lang/rust/issues/27812> for more information
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
extern crate rustc_query_system;
//~^ use of unstable library feature 'rustc_private'
//~| NOTE: see issue #27812 <https://github.com/rust-lang/rust/issues/27812> for more information
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date

use rustc_macros::HashStable;
//~^ use of unstable library feature 'rustc_private'
//~| NOTE: see issue #27812 <https://github.com/rust-lang/rust/issues/27812> for more information
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date

#[derive(HashStable)]
//~^ use of unstable library feature 'rustc_private'
//~| NOTE: in this expansion of #[derive(HashStable)]
//~| NOTE: in this expansion of #[derive(HashStable)]
//~| NOTE: in this expansion of #[derive(HashStable)]
//~| NOTE: in this expansion of #[derive(HashStable)]
//~| NOTE: see issue #27812 <https://github.com/rust-lang/rust/issues/27812> for more information
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
struct Test;

fn main() {}
