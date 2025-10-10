use crate::marker::PhantomData;
use crate::{mem, slice};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct IoSlice<'a> {
    vec: libc::iovec,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoSlice<'a> {
        IoSlice {
            vec: libc::iovec { iov_base: buf.as_ptr().cast_mut().cast(), iov_len: buf.len() },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        if self.vec.iov_len < n {
            panic!("advancing IoSlice beyond its length");
        }

        unsafe {
            self.vec.iov_len -= n;
            self.vec.iov_base = self.vec.iov_base.add(n);
        }
    }

    #[inline]
    pub const fn as_slice(&self) -> &'a [u8] {
        unsafe { slice::from_raw_parts(self.vec.iov_base as *const u8, self.vec.iov_len) }
    }

    #[cfg(target_env = "p1")]
    pub(crate) fn as_wasip1_slice<'b>(a: &'b [crate::io::IoSlice<'_>]) -> &'b [wasi::Ciovec] {
        let a = Self::as_libc_slice(a);

        assert_eq!(size_of::<wasi::Ciovec>(), size_of::<libc::iovec>());
        assert_eq!(align_of::<wasi::Ciovec>(), align_of::<libc::iovec>());
        assert_eq!(mem::offset_of!(wasi::Ciovec, buf), mem::offset_of!(libc::iovec, iov_base));
        assert_eq!(mem::offset_of!(wasi::Ciovec, buf_len), mem::offset_of!(libc::iovec, iov_len));

        // SAFETY: `wasi::Ciovec` and `libc::iovec` have different definitions
        // but have the same layout by definition, so it should be safe to
        // transmute between the two.
        unsafe { mem::transmute(a) }
    }

    pub(crate) fn as_libc_slice<'b>(a: &'b [crate::io::IoSlice<'_>]) -> &'b [libc::iovec] {
        assert_eq!(size_of::<IoSlice<'_>>(), size_of::<libc::iovec>());
        assert_eq!(align_of::<IoSlice<'_>>(), align_of::<libc::iovec>());

        // SAFETY: the `crate::io::IoSlice` type is a `repr(transparent)`
        // wrapper around `Self`, and `Self` is a `repr(transparent)` wrapper
        // aruond `libc::iovec`, thus an slice of one is a slice of the other.
        unsafe { mem::transmute(a) }
    }
}

#[repr(transparent)]
pub struct IoSliceMut<'a> {
    vec: libc::iovec,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        IoSliceMut {
            vec: libc::iovec { iov_base: buf.as_mut_ptr().cast(), iov_len: buf.len() },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        if self.vec.iov_len < n {
            panic!("advancing IoSlice beyond its length");
        }

        unsafe {
            self.vec.iov_len -= n;
            self.vec.iov_base = self.vec.iov_base.add(n);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.vec.iov_base as *const u8, self.vec.iov_len) }
    }

    #[inline]
    pub const fn into_slice(self) -> &'a mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.vec.iov_base as *mut u8, self.vec.iov_len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.vec.iov_base as *mut u8, self.vec.iov_len) }
    }

    #[cfg(target_env = "p1")]
    pub(crate) fn as_wasip1_slice<'b>(a: &'b mut [crate::io::IoSliceMut<'_>]) -> &'b [wasi::Iovec] {
        let a = Self::as_libc_slice(a);

        assert_eq!(size_of::<wasi::Iovec>(), size_of::<libc::iovec>());
        assert_eq!(align_of::<wasi::Iovec>(), align_of::<libc::iovec>());
        assert_eq!(mem::offset_of!(wasi::Iovec, buf), mem::offset_of!(libc::iovec, iov_base));
        assert_eq!(mem::offset_of!(wasi::Iovec, buf_len), mem::offset_of!(libc::iovec, iov_len));

        // SAFETY: `wasi::Iovec` and `libc::iovec` have different definitions
        // but have the same layout by definition, so it should be safe to
        // transmute between the two.
        unsafe { mem::transmute(a) }
    }

    pub(crate) fn as_libc_slice<'b>(a: &'b mut [crate::io::IoSliceMut<'_>]) -> &'b [libc::iovec] {
        assert_eq!(size_of::<IoSliceMut<'_>>(), size_of::<libc::iovec>());
        assert_eq!(align_of::<IoSliceMut<'_>>(), align_of::<libc::iovec>());

        // SAFETY: the `crate::io::IoSliceMut` type is a `repr(transparent)`
        // wrapper around `Self`, and `Self` is a `repr(transparent)` wrapper
        // aruond `libc::iovec`, thus an slice of one is a slice of the other.
        unsafe { mem::transmute(a) }
    }
}
