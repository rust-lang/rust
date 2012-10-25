//! Operations on unique pointer types

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

impl<T:Eq> ~const T : Eq {
    pure fn eq(other: &~const T) -> bool { *self == *(*other) }
    pure fn ne(other: &~const T) -> bool { *self != *(*other) }
}

impl<T:Ord> ~const T : Ord {
    pure fn lt(other: &~const T) -> bool { *self < *(*other) }
    pure fn le(other: &~const T) -> bool { *self <= *(*other) }
    pure fn ge(other: &~const T) -> bool { *self >= *(*other) }
    pure fn gt(other: &~const T) -> bool { *self > *(*other) }
}

