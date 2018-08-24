// Make sure that the spans of import errors are correct.

use abc::one_el;
//~^ ERROR 13:5: 13:8
use abc::{a, bbb, cccccc};
//~^ ERROR 15:5: 15:8
use a_very_long_name::{el, el2};
//~^ ERROR 17:5: 17:21

fn main() {}
