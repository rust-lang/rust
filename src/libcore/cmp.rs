/*!

The `Ord` and `Eq` comparison traits

This module contains the definition of both `Ord` and `Eq` which define
the common interfaces for doing comparison. Both are language items
that the compiler uses to implement the comparison operators. Rust code
may implement `Ord` to overload the `<`, `<=`, `>`, and `>=` operators,
and `Eq` to overload the `==` and `!=` operators.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

pub use nounittest::*;
pub use unittest::*;

/// Interfaces used for comparison.

// Awful hack to work around duplicate lang items in core test.
#[cfg(notest)]
mod nounittest {
    /**
     * Trait for values that can be compared for a sort-order.
     *
     * Eventually this may be simplified to only require
     * an `le` method, with the others generated from
     * default implementations.
     */
    #[lang="ord"]
    #[cfg(stage0)]
    pub trait Ord {
        pure fn lt(other: &self) -> bool;
        pure fn le(other: &self) -> bool;
        pure fn ge(other: &self) -> bool;
        pure fn gt(other: &self) -> bool;
    }

    #[lang="ord"]
    #[cfg(stage1)]
    #[cfg(stage2)]
    pub trait Ord {
        pure fn lt(&self, other: &self) -> bool;
        pure fn le(&self, other: &self) -> bool;
        pure fn ge(&self, other: &self) -> bool;
        pure fn gt(&self, other: &self) -> bool;
    }

    #[lang="eq"]
    /**
     * Trait for values that can be compared for equality
     * and inequality.
     *
     * Eventually this may be simplified to only require
     * an `eq` method, with the other generated from
     * a default implementation.
     */
    #[lang="eq"]
    #[cfg(stage0)]
    pub trait Eq {
        pure fn eq(other: &self) -> bool;
        pure fn ne(other: &self) -> bool;
    }

    #[lang="eq"]
    #[cfg(stage1)]
    #[cfg(stage2)]
    pub trait Eq {
        pure fn eq(&self, other: &self) -> bool;
        pure fn ne(&self, other: &self) -> bool;
    }
}

#[cfg(test)]
mod nounittest {
    #[legacy_exports];}

#[cfg(test)]
mod unittest {
    #[legacy_exports];

    #[cfg(stage0)]
    pub trait Ord {
        pure fn lt(other: &self) -> bool;
        pure fn le(other: &self) -> bool;
        pure fn ge(other: &self) -> bool;
        pure fn gt(other: &self) -> bool;
    }

    #[cfg(stage1)]
    #[cfg(stage2)]
    pub trait Ord {
        pure fn lt(&self, other: &self) -> bool;
        pure fn le(&self, other: &self) -> bool;
        pure fn ge(&self, other: &self) -> bool;
        pure fn gt(&self, other: &self) -> bool;
    }

    #[cfg(stage0)]
    pub trait Eq {
        pure fn eq(other: &self) -> bool;
        pure fn ne(other: &self) -> bool;
    }

    #[cfg(stage1)]
    #[cfg(stage2)]
    pub trait Eq {
        pure fn eq(&self, other: &self) -> bool;
        pure fn ne(&self, other: &self) -> bool;
    }
}

#[cfg(notest)]
mod unittest {
    #[legacy_exports];}

pub pure fn lt<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).lt(v2)
}

pub pure fn le<T: Ord Eq>(v1: &T, v2: &T) -> bool {
    (*v1).lt(v2) || (*v1).eq(v2)
}

pub pure fn eq<T: Eq>(v1: &T, v2: &T) -> bool {
    (*v1).eq(v2)
}

pub pure fn ne<T: Eq>(v1: &T, v2: &T) -> bool {
    (*v1).ne(v2)
}

pub pure fn ge<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).ge(v2)
}

pub pure fn gt<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).gt(v2)
}

