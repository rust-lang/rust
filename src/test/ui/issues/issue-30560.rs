type Alias = ();
use Alias::*;
//~^ ERROR unresolved import `Alias` [E0432]
//~| not a module `Alias`
use std::io::Result::*;
//~^ ERROR unresolved import `std::io::Result` [E0432]
//~| not a module `Result`

trait T {}
use T::*; //~ ERROR items in traits are not importable

fn main() {}
