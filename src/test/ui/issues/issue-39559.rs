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
    //~^ ERROR no function or associated item named `dim` found for type `D` in the current scope
    _dummy: D,
}

fn main() {}
