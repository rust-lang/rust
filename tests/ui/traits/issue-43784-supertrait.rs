pub trait Partial: Copy {
}

pub trait Complete: Partial {
}

impl<T> Partial for T where T: Complete {}
impl<T> Complete for T {} //~ ERROR trait `Copy` is not implemented for `T`

fn main() {}
