/// The underlying OsString/OsStr implementation on Windows is a
/// wrapper around the "WTF-8" encoding; see the `wtf8` module for more.

use crate::borrow::Cow;
use crate::fmt;
use crate::sys_common::wtf8::{self, Wtf8, Wtf8Buf};
use crate::mem;
use crate::rc::Rc;
use crate::sync::Arc;
use crate::ops::{Index, Range, RangeFrom, RangeTo};
use crate::sys_common::{AsInner, IntoInner, FromInner};
use core::slice::needles::{SliceSearcher, NaiveSearcher};
use crate::needle::Hay;

#[derive(Clone, Hash)]
pub struct Buf {
    pub inner: Wtf8Buf
}

impl IntoInner<Wtf8Buf> for Buf {
    fn into_inner(self) -> Wtf8Buf {
        self.inner
    }
}

impl FromInner<Wtf8Buf> for Buf {
    fn from_inner(inner: Wtf8Buf) -> Self {
        Buf { inner }
    }
}

impl AsInner<Wtf8> for Buf {
    fn as_inner(&self) -> &Wtf8 {
        &self.inner
    }
}

impl fmt::Debug for Buf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), formatter)
    }
}

impl fmt::Display for Buf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), formatter)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Slice {
    pub inner: Wtf8
}

impl fmt::Debug for Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl Index<Range<usize>> for Slice {
    type Output = Slice;

    fn index(&self, range: Range<usize>) -> &Slice {
        unsafe { mem::transmute(&self.inner[range]) }
    }
}

impl Index<RangeFrom<usize>> for Slice {
    type Output = Slice;

    fn index(&self, range: RangeFrom<usize>) -> &Slice {
        unsafe { mem::transmute(&self.inner[range]) }
    }
}

impl Index<RangeTo<usize>> for Slice {
    type Output = Slice;

    fn index(&self, range: RangeTo<usize>) -> &Slice {
        unsafe { mem::transmute(&self.inner[range]) }
    }
}

impl Buf {
    pub fn with_capacity(capacity: usize) -> Buf {
        Buf {
            inner: Wtf8Buf::with_capacity(capacity)
        }
    }

    pub fn clear(&mut self) {
        self.inner.clear()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn from_string(s: String) -> Buf {
        Buf { inner: Wtf8Buf::from_string(s) }
    }

    pub fn as_slice(&self) -> &Slice {
        unsafe { mem::transmute(self.inner.as_slice()) }
    }

    pub fn into_string(self) -> Result<String, Buf> {
        self.inner.into_string().map_err(|buf| Buf { inner: buf })
    }

    pub fn push_slice(&mut self, s: &Slice) {
        self.inner.push_wtf8(&s.inner)
    }

    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }

    #[inline]
    pub fn into_box(self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_box()) }
    }

    #[inline]
    pub fn from_box(boxed: Box<Slice>) -> Buf {
        let inner: Box<Wtf8> = unsafe { mem::transmute(boxed) };
        Buf { inner: Wtf8Buf::from_box(inner) }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        self.as_slice().into_arc()
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        self.as_slice().into_rc()
    }
}

impl Slice {
    pub fn from_str(s: &str) -> &Slice {
        unsafe { mem::transmute(Wtf8::from_str(s)) }
    }

    pub fn to_str(&self) -> Option<&str> {
        self.inner.as_str()
    }

    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.inner.to_string_lossy()
    }

    pub fn to_owned(&self) -> Buf {
        let mut buf = Wtf8Buf::with_capacity(self.inner.len());
        buf.push_wtf8(&self.inner);
        Buf { inner: buf }
    }

    #[inline]
    pub fn into_box(&self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_box()) }
    }

    pub fn empty_box() -> Box<Slice> {
        unsafe { mem::transmute(Wtf8::empty_box()) }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        let arc = self.inner.into_arc();
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Slice) }
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        let rc = self.inner.into_rc();
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const Slice) }
    }

    pub unsafe fn next_index(&self, index: usize) -> usize {
        self.inner.next_index(index)
    }

    pub unsafe fn prev_index(&self, index: usize) -> usize {
        self.inner.prev_index(index)
    }

    pub fn into_searcher(&self) -> InnerSearcher<SliceSearcher<'_, u8>> {
        wtf8::new_wtf8_searcher(&self.inner)
    }

    pub fn into_consumer(&self) -> InnerSearcher<NaiveSearcher<'_, u8>> {
        wtf8::new_wtf8_consumer(&self.inner)
    }

    pub fn as_bytes_for_searcher(&self) -> &[u8] {
        self.inner.as_inner()
    }
}

pub use crate::sys_common::wtf8::Wtf8Searcher as InnerSearcher;
