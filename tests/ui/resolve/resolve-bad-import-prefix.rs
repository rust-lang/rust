mod m {}
enum E {}
struct S;
trait Tr {}

use {}; // OK
use ::{}; // OK
use m::{}; // OK
use E::{}; // OK
use S::{}; // FIXME, this and `use S::{self};` should be an error
use Tr::{}; // FIXME, this and `use Tr::{self};` should be an error
use Nonexistent::{}; //~ ERROR unresolved import `Nonexistent`

fn main () {}
