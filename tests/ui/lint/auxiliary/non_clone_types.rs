pub struct NotClone;

pub struct IsClone;

impl Clone for IsClone {
    fn clone(&self) -> Self {
        Self
    }
}

pub struct ConditionalClone<T>(T);

impl<T: Clone> Clone for ConditionalClone<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

pub struct DifferentlyConditionalClone<T>(T);

impl<T: Default> Clone for DifferentlyConditionalClone<T> {
    fn clone(&self) -> Self {
        Self(T::default())
    }
}
