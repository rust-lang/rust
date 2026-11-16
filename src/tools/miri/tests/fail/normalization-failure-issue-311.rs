//@ normalize-stderr-test: "(\n)\n+$" -> "$1"
//@compile-flags: -Znext-solver

// The blanket impl says Item is u8 but the trait object says it's u16.
// Miri should report the normalization failure.
trait Project {
    type Item;
}

impl<T: ?Sized> Project for T {
    type Item = u8;
}

type Conflict = dyn Project<Item = u16>;

#[inline(never)]
fn projected_size<T: ?Sized + Project>() -> usize {
    std::mem::size_of::<T::Item>() //~ ERROR: post-monomorphization error
}

fn main() {
    std::hint::black_box(projected_size::<Conflict>());
}
