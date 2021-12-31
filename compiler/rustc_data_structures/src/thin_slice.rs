use std::ops::{Deref, DerefMut};
use thin_slice::ThinBoxedSlice;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct ThinSlice<T> {
    slice: ThinBoxedSlice<T>,
}

impl<T> ThinSlice<T> {
    pub fn into_vec(self) -> Vec<T> {
        self.into()
    }
}

impl<T> Default for ThinSlice<T> {
    fn default() -> Self {
        Self { slice: Default::default() }
    }
}

impl<T> From<Vec<T>> for ThinSlice<T> {
    fn from(vec: Vec<T>) -> Self {
        Self { slice: vec.into_boxed_slice2().into() }
    }
}

impl<T> From<ThinSlice<T>> for Vec<T> {
    fn from(slice: ThinSlice<T>) -> Self {
        let boxed: Box<[T]> = slice.slice.into();
        boxed.into_vec()
    }
}

impl<T> FromIterator<T> for ThinSlice<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let vec: Vec<T> = iter.into_iter().collect();
        vec.into()
    }
}

impl<T> Deref for ThinSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.slice.deref()
    }
}

impl<T> DerefMut for ThinSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice.deref_mut()
    }
}
