use crate::mem;

#[derive(Copy, Clone)]
pub struct IoSlice<'a>(&'a [u8]);

impl<'a> IoSlice<'a> {
    #[inline]
    pub const fn new(buf: &'a [u8]) -> IoSlice<'a> {
        IoSlice(buf)
    }

    #[inline]
    pub const fn advance(&mut self, n: usize) {
        let (_, remaining) = self.0.split_at(n);
        self.0 = remaining;
    }

    #[inline]
    pub const fn as_slice(&self) -> &'a [u8] {
        self.0
    }
}

pub struct IoSliceMut<'a>(&'a mut [u8]);

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub const fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        IoSliceMut(buf)
    }

    #[inline]
    pub const fn advance(&mut self, n: usize) {
        let slice = mem::replace(&mut self.0, &mut []);
        let (_, remaining) = slice.split_at_mut(n);
        self.0 = remaining;
    }

    #[inline]
    pub const fn as_slice(&self) -> &[u8] {
        self.0
    }

    #[inline]
    pub const fn into_slice(self) -> &'a mut [u8] {
        self.0
    }

    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        self.0
    }
}
