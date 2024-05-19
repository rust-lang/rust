// Do not suggest referencing the parameter to `check`

trait Marker<T> {}

impl<T> Marker<i32> for T {}

pub fn check<T: Marker<u32>>(_: T) {}

pub fn main() {
    check::<()>(()); //~ ERROR [E0277]
}
