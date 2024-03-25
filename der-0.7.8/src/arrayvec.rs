//! Array-backed append-only vector type.
// TODO(tarcieri): use `core` impl of `ArrayVec`
// See: https://github.com/rust-lang/rfcs/pull/2990

use crate::{ErrorKind, Result};

/// Array-backed append-only vector type.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) struct ArrayVec<T, const N: usize> {
    /// Elements of the set.
    elements: [Option<T>; N],

    /// Last populated element.
    length: usize,
}

impl<T, const N: usize> ArrayVec<T, N> {
    /// Create a new [`ArrayVec`].
    pub fn new() -> Self {
        Self {
            elements: [(); N].map(|_| None),
            length: 0,
        }
    }

    /// Push an item into this [`ArrayVec`].
    pub fn push(&mut self, item: T) -> Result<()> {
        match self.length.checked_add(1) {
            Some(n) if n <= N => {
                self.elements[self.length] = Some(item);
                self.length = n;
                Ok(())
            }
            _ => Err(ErrorKind::Overlength.into()),
        }
    }

    /// Get an element from this [`ArrayVec`].
    pub fn get(&self, index: usize) -> Option<&T> {
        match self.elements.get(index) {
            Some(Some(ref item)) => Some(item),
            _ => None,
        }
    }

    /// Iterate over the elements in this [`ArrayVec`].
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(&self.elements)
    }

    /// Is this [`ArrayVec`] empty?
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the number of elements in this [`ArrayVec`].
    pub fn len(&self) -> usize {
        self.length
    }

    /// Get the last item from this [`ArrayVec`].
    pub fn last(&self) -> Option<&T> {
        self.length.checked_sub(1).and_then(|n| self.get(n))
    }

    /// Extract the inner array.
    pub fn into_array(self) -> [Option<T>; N] {
        self.elements
    }
}

impl<T, const N: usize> AsRef<[Option<T>]> for ArrayVec<T, N> {
    fn as_ref(&self) -> &[Option<T>] {
        &self.elements[..self.length]
    }
}

impl<T, const N: usize> AsMut<[Option<T>]> for ArrayVec<T, N> {
    fn as_mut(&mut self) -> &mut [Option<T>] {
        &mut self.elements[..self.length]
    }
}

impl<T, const N: usize> Default for ArrayVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over the elements of an [`ArrayVec`].
#[derive(Clone, Debug)]
pub struct Iter<'a, T> {
    /// Decoder which iterates over the elements of the message.
    elements: &'a [Option<T>],

    /// Position within the iterator.
    position: usize,
}

impl<'a, T> Iter<'a, T> {
    pub(crate) fn new(elements: &'a [Option<T>]) -> Self {
        Self {
            elements,
            position: 0,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        match self.elements.get(self.position) {
            Some(Some(res)) => {
                self.position = self.position.checked_add(1)?;
                Some(res)
            }
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.elements.len().saturating_sub(self.position);
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

#[cfg(test)]
mod tests {
    use super::ArrayVec;
    use crate::ErrorKind;

    #[test]
    fn add() {
        let mut vec = ArrayVec::<u8, 3>::new();
        vec.push(1).unwrap();
        vec.push(2).unwrap();
        vec.push(3).unwrap();

        assert_eq!(vec.push(4).err().unwrap(), ErrorKind::Overlength.into());
        assert_eq!(vec.len(), 3);
    }
}
