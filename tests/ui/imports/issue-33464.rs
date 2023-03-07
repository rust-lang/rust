// Make sure that the spans of import errors are correct.

use abc::one_el;
//~^ ERROR
use abc::{a, bbb, cccccc};
//~^ ERROR
use a_very_long_name::{el, el2};
//~^ ERROR

fn main() {}
