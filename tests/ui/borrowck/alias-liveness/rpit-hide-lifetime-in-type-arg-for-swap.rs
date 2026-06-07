//@ revisions: edition2015 edition2024 polonius_alpha
//@ ignore-compare-mode-polonius (explicit revisions)
//@ [edition2015] edition: 2015
//@ [edition2024] edition: 2024
//@ [polonius_alpha] edition: 2024
//@ [polonius_alpha] compile-flags: -Zpolonius=next
//@ [polonius_alpha] known-bug: #153215
//@ [polonius_alpha] check-pass

// Like rpit-hide-lifetime-for-swap, but the invariant lifetime is hidden
// inside a *type* argument of the opaque rather than being captured as a
// region argument. The hidden type can capture `U` (and thus all of its
// regions) because of the `U: 'a` bound, so the regions inside the type
// argument must be considered live too.

trait Swap: Sized {
    fn swap(self, other: Self);
}

impl<T> Swap for &mut T {
    fn swap(self, other: Self) {
        std::mem::swap(self, other);
    }
}

fn hide<'a, U: Swap + 'a>(_: &'a u8, x: U) -> impl Swap + 'a {
    x
}

fn dangle() -> &'static [i32; 3] {
    let lock = 0u8;
    let mut res = &[4, 5, 6];
    let x = [1, 2, 3];
    hide(&lock, &mut res).swap(hide(&lock, &mut &x));
    res //[edition2015,edition2024]~ ERROR cannot return value referencing local variable `x`
}

fn main() {
    println!("{:?}", dangle());
}
