//@ revisions: edition2015 edition2024 polonius_alpha
//@ ignore-compare-mode-polonius (explicit revisions)
//@ [edition2015] edition: 2015
//@ [edition2024] edition: 2024
//@ [edition2015] compile-flags: -Zpolonius=nll
//@ [edition2024] compile-flags: -Zpolonius=nll
//@ [polonius_alpha] edition: 2024
//@ [polonius_alpha] compile-flags: -Zpolonius=next

// Like gat-hide-lifetime-for-swap, but the invariant lifetime is hidden
// inside a *type* argument of the GAT rather than being a region argument.
// The underlying type can capture `U` (and thus all of its regions) because
// of the declared `U: 'a` bound, so the regions inside the type argument
// must be considered live too.

trait Swap: Sized {
    fn swap(self, other: Self);
}
impl<T> Swap for &mut T {
    fn swap(self, other: Self) {
        std::mem::swap(self, other);
    }
}
trait Hider {
    type Hidden<'a, U>: Swap + 'a
    where
        Self: 'a,
        U: Swap + 'a;
    fn hide<'a, U: Swap + 'a>(&'a self, x: U) -> Self::Hidden<'a, U>;
}
fn dangle<H: Hider>(h: &H) -> &'static [i32; 3] {
    let mut res = &[4, 5, 6];
    let x = [1, 2, 3];
    h.hide(&mut res).swap(h.hide(&mut &x));
    res //[edition2015,edition2024,polonius_alpha]~ ERROR cannot return value referencing local variable `x`
}
struct H;
impl Hider for H {
    type Hidden<'a, U>
        = U
    where
        Self: 'a,
        U: Swap + 'a;
    fn hide<'a, U: Swap + 'a>(&'a self, x: U) -> Self::Hidden<'a, U> {
        x
    }
}
fn main() {
    println!("{:?}", dangle(&H));
}
