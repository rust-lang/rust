mod m {}
enum E {}
struct S;
trait Tr {}

use {}; // OK
use ::{}; // OK
use m::{}; // OK
use E::{}; // OK
use S::{}; //~ ERROR expected module or enum, found struct `S`
use Tr::{}; //~ ERROR expected module or enum, found trait `Tr`
use Nonexistent::{}; //~ ERROR cannot find module or enum `Nonexistent` in the crate root

fn main () {}
