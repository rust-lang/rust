//@ revisions: edition2015 edition2024 polonius_alpha
//@ ignore-compare-mode-polonius (explicit revisions)
//@ [edition2015] edition: 2015
//@ [edition2024] edition: 2024
//@ [edition2015] compile-flags: -Zpolonius=nll
//@ [edition2024] compile-flags: -Zpolonius=nll
//@ [polonius_alpha] edition: 2024
//@ [polonius_alpha] compile-flags: -Zpolonius=next

// Test to show what happens if we were not careful and allowed invariant
// lifetimes to escape though an impl trait.
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
    fn hide_ref<'a, 'b, T: 'static>(&self, x: &'a mut &'b T) -> impl Swap + 'a {
        x
    }
}
fn dangle_ref<H: Hider>(h: &H) -> &'static [i32; 3] {
    let mut res = &[4, 5, 6];
    let x = [1, 2, 3];
    h.hide_ref(&mut res).swap(h.hide_ref(&mut &x));
    res //[edition2015,edition2024,polonius_alpha]~ ERROR cannot return value referencing local variable `x`
}
struct H;
impl Hider for H {}

fn main() {
    println!("{:?}", dangle_ref(&H));
}
