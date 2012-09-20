/*!

Functions for the unit type.

*/

use cmp::{Eq, Ord};

#[cfg(stage0)]
impl () : Eq {
    pure fn eq(&&_other: ()) -> bool { true }
    pure fn ne(&&_other: ()) -> bool { false }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl () : Eq {
    pure fn eq(_other: &()) -> bool { true }
    pure fn ne(_other: &()) -> bool { false }
}

#[cfg(stage0)]
impl () : Ord {
    pure fn lt(&&_other: ()) -> bool { false }
    pure fn le(&&_other: ()) -> bool { true }
    pure fn ge(&&_other: ()) -> bool { true }
    pure fn gt(&&_other: ()) -> bool { false }
}
#[cfg(stage1)]
#[cfg(stage2)]
impl () : Ord {
    pure fn lt(_other: &()) -> bool { false }
    pure fn le(_other: &()) -> bool { true }
    pure fn ge(_other: &()) -> bool { true }
    pure fn gt(_other: &()) -> bool { false }
}

