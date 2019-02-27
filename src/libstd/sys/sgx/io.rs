pub struct IoVec<'a>(&'a [u8]);

impl<'a> IoVec<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoVec<'a> {
        IoVec(buf)
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.0
    }
}

pub struct IoVecMut<'a>(&'a mut [u8]);

impl<'a> IoVecMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoVecMut<'a> {
        IoVecMut(buf)
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.0
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.0
    }
}
