//! Implements `Eq` for vector types.

macro_rules! impl_eq {
    ($id:ident) => { impl Eq for $id {} }
}
