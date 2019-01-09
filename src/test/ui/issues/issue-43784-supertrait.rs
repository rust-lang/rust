pub trait Partial: Copy {
}

pub trait Complete: Partial {
}

impl<T> Partial for T where T: Complete {}
impl<T> Complete for T {} //~ ERROR the trait bound `T: std::marker::Copy` is not satisfied

fn main() {}
