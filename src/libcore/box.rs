//! Operations on shared box types

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

export ptr_eq;

pure fn ptr_eq<T>(a: @T, b: @T) -> bool {
    //! Determine if two shared boxes point to the same object
    unsafe { ptr::addr_of(*a) == ptr::addr_of(*b) }
}

impl<T:Eq> @const T : Eq {
    pure fn eq(&&other: @const T) -> bool { *self == *other }
    pure fn ne(&&other: @const T) -> bool { *self != *other }
}

impl<T:Ord> @const T : Ord {
    pure fn lt(&&other: @const T) -> bool { *self < *other }
    pure fn le(&&other: @const T) -> bool { *self <= *other }
    pure fn ge(&&other: @const T) -> bool { *self >= *other }
    pure fn gt(&&other: @const T) -> bool { *self > *other }
}

#[test]
fn test() {
    let x = @3;
    let y = @3;
    assert (ptr_eq::<int>(x, x));
    assert (ptr_eq::<int>(y, y));
    assert (!ptr_eq::<int>(x, y));
    assert (!ptr_eq::<int>(y, x));
}
