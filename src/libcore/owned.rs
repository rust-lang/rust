//! Operations on unique pointer types

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

impl<T:Eq> ~const T : Eq {
    pure fn eq(&self, other: &~const T) -> bool { *(*self) == *(*other) }
    pure fn ne(&self, other: &~const T) -> bool { *(*self) != *(*other) }
}

impl<T:Ord> ~const T : Ord {
    pure fn lt(&self, other: &~const T) -> bool { *(*self) < *(*other) }
    pure fn le(&self, other: &~const T) -> bool { *(*self) <= *(*other) }
    pure fn ge(&self, other: &~const T) -> bool { *(*self) >= *(*other) }
    pure fn gt(&self, other: &~const T) -> bool { *(*self) > *(*other) }
}

