// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

/// Interfaces used for comparison.

// Awful hack to work around duplicate lang items in core test.

/**
 * Trait for values that can be compared for a sort-order.
 *
 * Eventually this may be simplified to only require
 * an `le` method, with the others generated from
 * default implementations.
 */
#[cfg(notest)]
#[lang="ord"]
trait Ord {
    pure fn lt(&&other: self) -> bool;
    pure fn le(&&other: self) -> bool;
    pure fn ge(&&other: self) -> bool;
    pure fn gt(&&other: self) -> bool;
}

#[cfg(test)]
trait Ord {
    pure fn lt(&&other: self) -> bool;
    pure fn le(&&other: self) -> bool;
    pure fn ge(&&other: self) -> bool;
    pure fn gt(&&other: self) -> bool;
}

#[cfg(notest)]
#[lang="eq"]
/**
 * Trait for values that can be compared for equality
 * and inequality.
 *
 * Eventually this may be simplified to only require
 * an `eq` method, with the other generated from
 * a default implementation.
 */
trait Eq {
    pure fn eq(&&other: self) -> bool;
    pure fn ne(&&other: self) -> bool;
}

#[cfg(test)]
trait Eq {
    pure fn eq(&&other: self) -> bool;
    pure fn ne(&&other: self) -> bool;
}

pure fn lt<T: Ord>(v1: &T, v2: &T) -> bool {
    v1.lt(v2)
}

pure fn le<T: Ord Eq>(v1: &T, v2: &T) -> bool {
    v1.lt(v2) || v1.eq(v2)
}

pure fn eq<T: Eq>(v1: &T, v2: &T) -> bool {
    v1.eq(v2)
}

pure fn ne<T: Eq>(v1: &T, v2: &T) -> bool {
    v1.ne(v2)
}

pure fn ge<T: Ord>(v1: &T, v2: &T) -> bool {
    v1.ge(v2)
}

pure fn gt<T: Ord>(v1: &T, v2: &T) -> bool {
    v1.gt(v2)
}

