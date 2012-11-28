/*!

Functions for the unit type.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

impl () : Eq {
    pure fn eq(&self, _other: &()) -> bool { true }
    pure fn ne(&self, _other: &()) -> bool { false }
}

impl () : Ord {
    pure fn lt(&self, _other: &()) -> bool { false }
    pure fn le(&self, _other: &()) -> bool { true }
    pure fn ge(&self, _other: &()) -> bool { true }
    pure fn gt(&self, _other: &()) -> bool { false }
}

