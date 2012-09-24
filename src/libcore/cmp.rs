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

use nounittest::*;
use unittest::*;
export Ord;
export Eq;

/// Interfaces used for comparison.

// Awful hack to work around duplicate lang items in core test.
#[cfg(notest)]
mod nounittest {
    #[legacy_exports];
    /**
     * Trait for values that can be compared for a sort-order.
     *
     * Eventually this may be simplified to only require
     * an `le` method, with the others generated from
     * default implementations.
     */
    #[lang="ord"]
    trait Ord {
        pure fn lt(other: &self) -> bool;
        pure fn le(other: &self) -> bool;
        pure fn ge(other: &self) -> bool;
        pure fn gt(other: &self) -> bool;
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
    trait Eq {
        pure fn eq(other: &self) -> bool;
        pure fn ne(other: &self) -> bool;
    }
}

#[cfg(test)]
mod nounittest {
    #[legacy_exports];}

#[cfg(test)]
mod unittest {
    #[legacy_exports];
    trait Ord {
        pure fn lt(other: &self) -> bool;
        pure fn le(other: &self) -> bool;
        pure fn ge(other: &self) -> bool;
        pure fn gt(other: &self) -> bool;
    }

    trait Eq {
        pure fn eq(other: &self) -> bool;
        pure fn ne(other: &self) -> bool;
    }
}

#[cfg(notest)]
mod unittest {
    #[legacy_exports];}

pure fn lt<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).lt(v2)
}

pure fn le<T: Ord Eq>(v1: &T, v2: &T) -> bool {
    (*v1).lt(v2) || (*v1).eq(v2)
}

pure fn eq<T: Eq>(v1: &T, v2: &T) -> bool {
    (*v1).eq(v2)
}

pure fn ne<T: Eq>(v1: &T, v2: &T) -> bool {
    (*v1).ne(v2)
}

pure fn ge<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).ge(v2)
}

pure fn gt<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).gt(v2)
}

