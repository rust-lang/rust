//! A [`Vec`] that is guaranteed to at least contain one element.

pub struct NonEmptyVec<T>(Vec<T>);

impl<T> NonEmptyVec<T> {
    #[inline]
    pub fn new(initial: T) -> Self {
        NonEmptyVec(vec![initial])
    }

    #[inline]
    pub fn last_mut(&mut self) -> &mut T {
        match self.0.last_mut() {
            Some(it) => it,
            None => unreachable!(),
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.0.len() <= 1 {
            None
        } else {
            self.0.pop()
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        self.0.push(value)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn into_first(mut self) -> T {
        match self.0.pop() {
            Some(it) => it,
            None => unreachable!(),
        }
    }
}
