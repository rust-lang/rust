use std::ops::{Deref, DerefMut};
use std::slice;

use rustc_data_structures::smallvec::{self, SmallVec};
use rustc_span::Symbol;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ItemPath(SmallVec<[Symbol; 4]>);

impl From<SmallVec<[Symbol; 4]>> for ItemPath {
    fn from(value: SmallVec<[Symbol; 4]>) -> Self {
        Self(value)
    }
}

impl From<&[Symbol]> for ItemPath {
    fn from(value: &[Symbol]) -> Self {
        SmallVec::from_slice(value).into()
    }
}

impl FromIterator<Symbol> for ItemPath {
    fn from_iter<T: IntoIterator<Item = Symbol>>(iter: T) -> Self {
        Self(FromIterator::from_iter(iter))
    }
}

impl Deref for ItemPath {
    type Target = SmallVec<[Symbol; 4]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ItemPath {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for ItemPath {
    type Item = Symbol;

    type IntoIter = smallvec::IntoIter<[Symbol; 4]>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a ItemPath {
    type Item = &'a Symbol;

    type IntoIter = slice::Iter<'a, Symbol>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
