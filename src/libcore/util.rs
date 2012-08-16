// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

/**
 * Miscellaneous helpers for common patterns.
 */

/// The identity function.
pure fn id<T>(+x: T) -> T { x }

/// Ignores a value.
pure fn ignore<T>(+_x: T) { }

/**
 * Swap the values at two mutable locations of the same type, without
 * deinitialising or copying either one.
 */
#[inline(always)]
fn swap<T>(x: &mut T, y: &mut T) {
    *x <-> *y;
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline(always)]
fn replace<T>(dest: &mut T, +src: T) -> T {
    let mut tmp = src;
    swap(dest, &mut tmp);
    tmp
}

/// A non-copyable dummy type.
struct NonCopyable {
    i: ();
    new() { self.i = (); }
    drop { }
}

mod tests {
    #[test]
    fn identity_crisis() {
        // Writing a test for the identity function. How did it come to this?
        let x = ~[{mut a: 5, b: false}];
        assert x == id(copy x);
    }
    #[test]
    fn test_swap() {
        let mut x = 31337;
        let mut y = 42;
        swap(&mut x, &mut y);
        assert x == 42;
        assert y == 31337;
    }
    #[test]
    fn test_replace() {
        let mut x = some(NonCopyable());
        let y = replace(&mut x, none);
        assert x.is_none();
        assert y.is_some();
    }
}
