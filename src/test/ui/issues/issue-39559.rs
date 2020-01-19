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
    //~^ ERROR type parameters cannot appear within an array length expression [E0747]
    _dummy: D,
}

fn main() {}
