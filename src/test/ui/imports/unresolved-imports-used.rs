// There should be *one* unused import error.
#![deny(unused_imports)]

mod qux {
   fn quz() {}
   pub fn quy() {}
}

use qux::quz;  //~ ERROR function `quz` is private
use qux::bar;  //~ ERROR unresolved import `qux::bar`
use foo::bar;
use baz::*;
use qux::bar2; //~ ERROR unresolved import `qux::bar2`
use foo2::bar2;
use baz2::*;
use qux::quy;  //~ ERROR unused import

fn main() {}
