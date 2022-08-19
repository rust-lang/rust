//! The underlying OsString/OsStr implementation on UEFI systems
//! Just using standard UTF-8. Since UCS-2 can be represented in UTF-8, no need for using a
//! non-standard encoding.

use crate::borrow::Cow;
use crate::collections::TryReserveError;
use crate::fmt;
use crate::mem;
use crate::rc::Rc;
use crate::str;
use crate::sync::Arc;
use crate::sys_common::{AsInner, IntoInner};

#[derive(Hash)]
#[repr(transparent)]
pub struct Buf {
    pub inner: String,
}

#[repr(transparent)]
pub struct Slice {
    pub inner: str,
}

impl fmt::Debug for Slice {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This is safe since Slice is always valid UTF-8
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

impl fmt::Display for Slice {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This is safe since Slice is always valid UTF-8
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl fmt::Debug for Buf {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), formatter)
    }
}

impl fmt::Display for Buf {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), formatter)
    }
}

impl Clone for Buf {
    #[inline]
    fn clone(&self) -> Self {
        Buf { inner: self.inner.clone() }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner)
    }
}

impl IntoInner<String> for Buf {
    #[inline]
    fn into_inner(self) -> String {
        self.inner
    }
}

impl AsInner<str> for Buf {
    #[inline]
    fn as_inner(&self) -> &str {
        &self.inner
    }
}

impl Buf {
    #[inline]
    pub fn from_string(s: String) -> Buf {
        Buf { inner: s }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Buf {
        Buf { inner: String::with_capacity(capacity) }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve_exact(additional)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }

    #[inline]
    pub fn as_slice(&self) -> &Slice {
        // SAFETY: Slice just wraps str,
        // and &*self.inner is &str, therefore
        // transmuting &str to &Slice is safe.
        unsafe { mem::transmute(&*self.inner) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut Slice {
        // SAFETY: Slice just wraps str,
        // and &mut *self.inner is &mut str, therefore
        // transmuting &mut str to &mut Slice is safe.
        unsafe { mem::transmute(&mut *self.inner) }
    }

    #[inline]
    pub fn into_string(self) -> Result<String, Buf> {
        // This should never fail since OsString for UEFI is always valid UTF-8
        Ok(self.inner)
    }

    #[inline]
    pub fn push_slice(&mut self, s: &Slice) {
        self.inner.push_str(&s.inner)
    }

    #[inline]
    pub fn into_box(self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_boxed_str()) }
    }

    #[inline]
    pub fn from_box(boxed: Box<Slice>) -> Buf {
        let inner: Box<str> = unsafe { mem::transmute(boxed) };
        Buf { inner: String::from(inner) }
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
    #[inline]
    pub fn from_str(s: &str) -> &Slice {
        unsafe { mem::transmute(s) }
    }

    #[inline]
    pub fn to_str(&self) -> Option<&str> {
        // This is safe since OsStr for UEFI is always valid UTF-8
        Some(&self.inner)
    }

    #[inline]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        Cow::from(&self.inner)
    }

    #[inline]
    pub fn to_owned(&self) -> Buf {
        Buf { inner: self.inner.to_string() }
    }

    #[inline]
    pub fn clone_into(&self, buf: &mut Buf) {
        self.inner.clone_into(&mut buf.inner)
    }

    #[inline]
    pub fn into_box(&self) -> Box<Slice> {
        let boxed: Box<str> = self.inner.into();
        unsafe { mem::transmute(boxed) }
    }

    #[inline]
    pub fn empty_box() -> Box<Slice> {
        let boxed: Box<str> = Default::default();
        unsafe { mem::transmute(boxed) }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        let arc: Arc<str> = Arc::from(&self.inner);
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Slice) }
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        let rc: Rc<str> = Rc::from(&self.inner);
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const Slice) }
    }

    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        self.inner.make_ascii_lowercase()
    }

    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        self.inner.make_ascii_uppercase()
    }

    #[inline]
    pub fn to_ascii_lowercase(&self) -> Buf {
        Buf { inner: self.inner.to_ascii_lowercase() }
    }

    #[inline]
    pub fn to_ascii_uppercase(&self) -> Buf {
        Buf { inner: self.inner.to_ascii_uppercase() }
    }

    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.inner.is_ascii()
    }

    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &Self) -> bool {
        self.inner.eq_ignore_ascii_case(&other.inner)
    }
}
