use crate::mem;

#[derive(Copy, Clone)]
pub(super) struct IoSlice<'a>(&'a [u8]);

impl<'a> IoSlice<'a> {
    #[inline]
    pub(super) fn new(buf: &'a [u8]) -> IoSlice<'a> {
        IoSlice(buf)
    }

    #[inline]
    pub(super) fn advance(&mut self, n: usize) {
        self.0 = &self.0[n..]
    }

    #[inline]
    pub(super) const fn as_slice(&self) -> &'a [u8] {
        self.0
    }
}

pub(super) struct IoSliceMut<'a>(&'a mut [u8]);

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub(super) fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        IoSliceMut(buf)
    }

    #[inline]
    pub(super) fn advance(&mut self, n: usize) {
        let slice = mem::take(&mut self.0);
        let (_, remaining) = slice.split_at_mut(n);
        self.0 = remaining;
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[u8] {
        self.0
    }

    #[inline]
    pub(super) const fn into_slice(self) -> &'a mut [u8] {
        self.0
    }

    #[inline]
    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.0
    }
}
