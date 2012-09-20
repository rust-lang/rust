//! Operations on unique pointer types

use cmp::{Eq, Ord};

#[cfg(stage0)]
impl<T:Eq> ~const T : Eq {
    pure fn eq(&&other: ~const T) -> bool { *self == *other }
    pure fn ne(&&other: ~const T) -> bool { *self != *other }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl<T:Eq> ~const T : Eq {
    pure fn eq(other: &~const T) -> bool { *self == *(*other) }
    pure fn ne(other: &~const T) -> bool { *self != *(*other) }
}

#[cfg(stage0)]
impl<T:Ord> ~const T : Ord {
    pure fn lt(&&other: ~const T) -> bool { *self < *other }
    pure fn le(&&other: ~const T) -> bool { *self <= *other }
    pure fn ge(&&other: ~const T) -> bool { *self >= *other }
    pure fn gt(&&other: ~const T) -> bool { *self > *other }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl<T:Ord> ~const T : Ord {
    pure fn lt(other: &~const T) -> bool { *self < *(*other) }
    pure fn le(other: &~const T) -> bool { *self <= *(*other) }
    pure fn ge(other: &~const T) -> bool { *self >= *(*other) }
    pure fn gt(other: &~const T) -> bool { *self > *(*other) }
}

