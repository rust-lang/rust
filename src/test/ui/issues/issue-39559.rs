trait Dim {
    fn dim() -> usize;
}

enum Dim3 {}

impl Dim for Dim3 {
    fn dim() -> usize {
        3
    }
}

pub struct Vector<T, D: Dim> {
    entries: [T; D::dim()],
    //~^ ERROR type parameters can't appear within an array length expression [E0447]
    _dummy: D,
}

fn main() {}
