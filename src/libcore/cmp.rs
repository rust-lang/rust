// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

/// Interfaces used for comparison.

#[cfg(notest)]
#[lang="ord"]
trait Ord {
    pure fn lt(&&other: self) -> bool;
}

#[cfg(notest)]
#[lang="eq"]
trait Eq {
    pure fn eq(&&other: self) -> bool;
}

#[cfg(notest)]
pure fn lt<T: Ord>(v1: &T, v2: &T) -> bool {
    v1.lt(*v2)
}

#[cfg(notest)]
pure fn le<T: Ord Eq>(v1: &T, v2: &T) -> bool {
    v1.lt(*v2) || v1.eq(*v2)
}

#[cfg(notest)]
pure fn eq<T: Eq>(v1: &T, v2: &T) -> bool {
    v1.eq(*v2)
}
