// Just a named container for our op, so it can have impls
pub struct NominalOp<T> {
    op: T
}

impl<T: Copy> NominalOp<T>: Clone {
    fn clone(&self) -> NominalOp<T> { copy *self }
}

