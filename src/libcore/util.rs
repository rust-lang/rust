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

/// Sets `*ptr` to `new_value`, invokes `op()`, and then restores the
/// original value of `*ptr`.
#[inline(always)]
fn with<T: copy, R>(
    ptr: &mut T,
    +new_value: T,
    op: &fn() -> R) -> R
{
    // NDM: if swap operator were defined somewhat differently,
    // we wouldn't need to copy...

    let old_value = *ptr;
    *ptr = move new_value;
    let result = op();
    *ptr = move old_value;
    return move result;
}

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
        let x = ~[(5, false)];
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
        let mut x = Some(NonCopyable());
        let y = replace(&mut x, None);
        assert x.is_none();
        assert y.is_some();
    }
}
