// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

/// Interfaces used for comparison.

trait ord {
    pure fn lt(&&other: self) -> bool;
}

trait eq {
    pure fn eq(&&other: self) -> bool;
}

pure fn lt<T: ord>(v1: &T, v2: &T) -> bool {
    v1.lt(*v2)
}

pure fn le<T: ord eq>(v1: &T, v2: &T) -> bool {
    v1.lt(*v2) || v1.eq(*v2)
}

pure fn eq<T: eq>(v1: &T, v2: &T) -> bool {
    v1.eq(*v2)
}
