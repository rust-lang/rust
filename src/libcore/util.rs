/*!

Miscellaneous helpers for common patterns.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::Eq;

/// The identity function.
#[inline(always)]
pub pure fn id<T>(x: T) -> T { move x }

/// Ignores a value.
#[inline(always)]
pub pure fn ignore<T>(_x: T) { }

/// Sets `*ptr` to `new_value`, invokes `op()`, and then restores the
/// original value of `*ptr`.
#[inline(always)]
pub fn with<T: Copy, R>(
    ptr: &mut T,
    new_value: T,
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
pub fn swap<T>(x: &mut T, y: &mut T) {
    *x <-> *y;
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline(always)]
pub fn replace<T>(dest: &mut T, src: T) -> T {
    let mut tmp = move src;
    swap(dest, &mut tmp);
    move tmp
}

/// A non-copyable dummy type.
pub struct NonCopyable {
    i: (),
    drop { }
}

pub fn NonCopyable() -> NonCopyable { NonCopyable { i: () } }

/**
A utility function for indicating unreachable code. It will fail if
executed. This is occasionally useful to put after loops that never
terminate normally, but instead directly return from a function.

# Example

~~~
fn choose_weighted_item(v: &[Item]) -> Item {
    assert v.is_not_empty();
    let mut so_far = 0u;
    for v.each |item| {
        so_far += item.weight;
        if so_far > 100 {
            return item;
        }
    }
    // The above loop always returns, so we must hint to the
    // type checker that it isn't possible to get down here
    util::unreachable();
}
~~~

*/
pub fn unreachable() -> ! {
    fail ~"internal error: entered unreachable code";
}

mod tests {
    #[legacy_exports];
    #[test]
    fn identity_crisis() {
        // Writing a test for the identity function. How did it come to this?
        let x = ~[(5, false)];
        //FIXME #3387 assert x.eq(id(copy x));
        let y = copy x;
        assert x.eq(&id(move y));
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
