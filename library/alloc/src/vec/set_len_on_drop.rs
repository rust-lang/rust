// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
//
// The idea is: The length field in SetLenOnDrop is a local variable
// that the optimizer will see does not alias with any stores through the Vec's data
// pointer. This is a workaround for alias analysis issue #32155
pub(super) struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    pub(super) fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop { local_len: *len, len }
    }

    /// # Safety
    /// `self.current_len() + increment` must not overflow.
    #[inline]
    pub(super) unsafe fn increment_len_unchecked(&mut self, increment: usize) {
        // SAFETY: This is our precondition
        self.local_len = unsafe { self.local_len.unchecked_add(increment) };
    }

    #[inline]
    pub(super) fn current_len(&self) -> usize {
        self.local_len
    }
}

impl Drop for SetLenOnDrop<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
    }
}
