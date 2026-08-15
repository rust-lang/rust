//@ edition:2015
mod m {}
enum E {}
struct S;
trait Tr {}

use {}; // OK
use ::{}; // OK
use m::{}; // OK
use E::{}; // OK
use S::{}; //~ ERROR unresolved import `S`
use Tr::{}; // FIXME, this and `use Tr::{self};` should be an error
use Nonexistent::{}; //~ ERROR unresolved import `Nonexistent`

fn main () {}
