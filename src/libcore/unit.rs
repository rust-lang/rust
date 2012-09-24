/*!

Functions for the unit type.

*/

use cmp::{Eq, Ord};

impl () : Eq {
    pure fn eq(_other: &()) -> bool { true }
    pure fn ne(_other: &()) -> bool { false }
}

impl () : Ord {
    pure fn lt(_other: &()) -> bool { false }
    pure fn le(_other: &()) -> bool { true }
    pure fn ge(_other: &()) -> bool { true }
    pure fn gt(_other: &()) -> bool { false }
}

