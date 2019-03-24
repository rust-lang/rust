// compile-fail

// See https://github.com/rust-lang/rust/pull/57870#issuecomment-457333709 for more details.

trait Swap: Sized {
    fn swap(self, other: Self);
}

impl<T> Swap for &mut T {
    fn swap(self, other: Self) {
        std::mem::swap(self, other);
    }
}

// The hidden relifetimegion `'b` should not be allowed to escape this function, since it may not
// outlive '`a`, in which case we have a dangling reference.
fn hide<'a, 'b, T: 'static>(x: &'a mut &'b T) -> impl Swap + 'a {
//~^ lifetime mismatch [E0623]
    x
}

fn dangle() -> &'static [i32; 3] {
    let mut res = &[4, 5, 6];
    let x = [1, 2, 3];
    hide(&mut res).swap(hide(&mut &x));
    res
}

fn main() {}
