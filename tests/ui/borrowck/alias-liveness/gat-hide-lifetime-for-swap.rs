//@ revisions: edition2015 edition2024 polonius_alpha
//@ ignore-compare-mode-polonius (explicit revisions)
//@ [edition2015] edition: 2015
//@ [edition2024] edition: 2024
//@ [polonius_alpha] edition: 2024
//@ [polonius_alpha] compile-flags: -Zpolonius=next

// Test to show what happens if we were not careful and allowed invariant
// lifetimes to escape through a GAT.
//
// Specifically we swap a long lived and short lived reference, giving us a
// dangling pointer.

trait Swap: Sized {
    fn swap(self, other: Self);
}
impl<T> Swap for &mut T {
    fn swap(self, other: Self) {
        std::mem::swap(self, other);
    }
}
trait Hider {
    type Hidden<'a, 'b, T: 'static>: Swap + 'a
    where
        Self: 'a,
        'b: 'a;
    fn hide_ref<'a, 'b, T: 'static>(&self, x: &'a mut &'b T) -> Self::Hidden<'a, 'b, T>;
}
fn dangle_ref<H: Hider>(h: &H) -> &'static [i32; 3] {
    let mut res = &[4, 5, 6];
    let x = [1, 2, 3];
    h.hide_ref(&mut res).swap(h.hide_ref(&mut &x));
    res //[edition2015,edition2024,polonius_alpha]~ ERROR cannot return value referencing local variable `x`
}
struct H;
impl Hider for H {
    type Hidden<'a, 'b, T: 'static>
        = &'a mut &'b T
    where
        Self: 'a,
        'b: 'a;
    fn hide_ref<'a, 'b, T: 'static>(&self, x: &'a mut &'b T) -> Self::Hidden<'a, 'b, T> {
        x
    }
}
fn main() {
    println!("{:?}", dangle_ref(&H));
}
