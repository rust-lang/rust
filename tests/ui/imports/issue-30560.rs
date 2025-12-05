type Alias = ();
use Alias::*; //~ ERROR unresolved import `Alias` [E0432]

use std::io::Result::*; //~ ERROR unresolved import `std::io::Result` [E0432]

fn main() {}
