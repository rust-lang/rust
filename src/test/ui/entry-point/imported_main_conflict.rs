#![feature(imported_main)]
//~^ ERROR `main` is ambiguous (glob import vs glob import in the same module)
mod m1 { pub(crate) fn main() {} }
mod m2 { pub(crate) fn main() {} }

use m1::*;
use m2::*;
