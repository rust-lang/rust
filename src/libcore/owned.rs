//! Operations on unique pointer types

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

impl<T:Eq> ~const T : Eq {
    #[cfg(stage0)]
    pure fn eq(other: &~const T) -> bool { *self == *(*other) }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn eq(&self, other: &~const T) -> bool { *(*self) == *(*other) }
    #[cfg(stage0)]
    pure fn ne(other: &~const T) -> bool { *self != *(*other) }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn ne(&self, other: &~const T) -> bool { *(*self) != *(*other) }
}

impl<T:Ord> ~const T : Ord {
    #[cfg(stage0)]
    pure fn lt(other: &~const T) -> bool { *self < *(*other) }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn lt(&self, other: &~const T) -> bool { *(*self) < *(*other) }
    #[cfg(stage0)]
    pure fn le(other: &~const T) -> bool { *self <= *(*other) }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn le(&self, other: &~const T) -> bool { *(*self) <= *(*other) }
    #[cfg(stage0)]
    pure fn ge(other: &~const T) -> bool { *self >= *(*other) }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn ge(&self, other: &~const T) -> bool { *(*self) >= *(*other) }
    #[cfg(stage0)]
    pure fn gt(other: &~const T) -> bool { *self > *(*other) }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn gt(&self, other: &~const T) -> bool { *(*self) > *(*other) }
}

