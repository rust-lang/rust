//! Operations on unique pointer types

import cmp::{Eq, Ord};

impl<T:Eq> ~const T : Eq {
    pure fn eq(&&other: ~const T) -> bool { *self == *other }
}

impl<T:Ord> ~const T : Ord {
    pure fn lt(&&other: ~const T) -> bool { *self < *other }
}

