/**
 * Miscellaneous helpers for common patterns.
 */

/**
 * Swap the values at two mutable locations of the same type, without
 * deinitialising or copying either one.
 */
fn swap<T>(x: &mut T, y: &mut T) {
    *x <-> *y;
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
fn replace<T>(dest: &mut T, +src: T) -> T {
    let mut tmp = src;
    swap(dest, &mut tmp);
    tmp
}

/// A non-copyable dummy type.
class noncopyable {
    i: ();
    new() { self.i = (); }
    drop { }
}

mod tests {
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
        let mut x = some(noncopyable());
        let y = replace(&mut x, none);
        assert x.is_none();
        assert y.is_some();
    }
}
