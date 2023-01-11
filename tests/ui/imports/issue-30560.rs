type Alias = ();
use Alias::*; //~ ERROR unresolved import `Alias` [E0432]

use std::io::Result::*; //~ ERROR unresolved import `std::io::Result` [E0432]

trait T {}
use T::*; //~ ERROR items in traits are not importable

fn main() {}
