/*!

Functions for the unit type.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};

impl () : Eq {
    #[cfg(stage0)]
    pure fn eq(_other: &()) -> bool { true }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn eq(&self, _other: &()) -> bool { true }
    #[cfg(stage0)]
    pure fn ne(_other: &()) -> bool { false }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn ne(&self, _other: &()) -> bool { false }
}

impl () : Ord {
    #[cfg(stage0)]
    pure fn lt(_other: &()) -> bool { false }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn lt(&self, _other: &()) -> bool { false }
    #[cfg(stage0)]
    pure fn le(_other: &()) -> bool { true }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn le(&self, _other: &()) -> bool { true }
    #[cfg(stage0)]
    pure fn ge(_other: &()) -> bool { true }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn ge(&self, _other: &()) -> bool { true }
    #[cfg(stage0)]
    pure fn gt(_other: &()) -> bool { false }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn gt(&self, _other: &()) -> bool { false }
}

