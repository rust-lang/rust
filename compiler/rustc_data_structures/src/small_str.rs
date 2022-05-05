use smallvec::SmallVec;

#[cfg(test)]
mod tests;

/// Like SmallVec but for strings.
#[derive(Default)]
pub struct SmallStr<const N: usize>(SmallVec<[u8; N]>);

impl<const N: usize> SmallStr<N> {
    #[inline]
    pub fn new() -> Self {
        SmallStr(SmallVec::default())
    }

    #[inline]
    pub fn push_str(&mut self, s: &str) {
        self.0.extend_from_slice(s.as_bytes());
    }

    #[inline]
    pub fn empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn spilled(&self) -> bool {
        self.0.spilled()
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.0.as_slice()) }
    }
}

impl<const N: usize> std::ops::Deref for SmallStr<N> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize, A: AsRef<str>> FromIterator<A> for SmallStr<N> {
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = A>,
    {
        let mut s = SmallStr::default();
        s.extend(iter);
        s
    }
}

impl<const N: usize, A: AsRef<str>> Extend<A> for SmallStr<N> {
    #[inline]
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = A>,
    {
        for a in iter.into_iter() {
            self.push_str(a.as_ref());
        }
    }
}
