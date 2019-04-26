#![deny(unused_imports)]
// there should be *no* unused import errors

mod qux {
   pub(in crate::qux) fn quz() {}
}

use qux::quz;//~ ERROR function `quz` is private
use qux::bar;//~ ERROR unresolved import `qux::bar`
use foo::bar; //~ ERROR unresolved import `foo`

fn main() {}
