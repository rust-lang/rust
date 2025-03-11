//! Trait implementations for `ByteStr`.

use crate::bstr::ByteStr;
use crate::cmp::Ordering;
use crate::hash;
use crate::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

#[unstable(feature = "bstr", issue = "134915")]
impl Ord for ByteStr {
    #[inline]
    fn cmp(&self, other: &ByteStr) -> Ordering {
        Ord::cmp(&self.0, &other.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl PartialOrd for ByteStr {
    #[inline]
    fn partial_cmp(&self, other: &ByteStr) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.0, &other.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl PartialEq<ByteStr> for ByteStr {
    #[inline]
    fn eq(&self, other: &ByteStr) -> bool {
        &self.0 == &other.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Eq for ByteStr {}

#[unstable(feature = "bstr", issue = "134915")]
impl hash::Hash for ByteStr {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[doc(hidden)]
#[macro_export]
#[unstable(feature = "bstr_internals", issue = "none")]
macro_rules! impl_partial_eq {
    ($lhs:ty, $rhs:ty) => {
        #[allow(unused_lifetimes)]
        impl<'a> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                let other: &[u8] = other.as_ref();
                PartialEq::eq(self.as_bytes(), other)
            }
        }

        #[allow(unused_lifetimes)]
        impl<'a> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                let this: &[u8] = self.as_ref();
                PartialEq::eq(this, other.as_bytes())
            }
        }
    };
}

#[doc(hidden)]
#[unstable(feature = "bstr_internals", issue = "none")]
pub use impl_partial_eq;

#[doc(hidden)]
#[macro_export]
#[unstable(feature = "bstr_internals", issue = "none")]
macro_rules! impl_partial_eq_ord {
    ($lhs:ty, $rhs:ty) => {
        $crate::bstr::impl_partial_eq!($lhs, $rhs);

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<'a> PartialOrd<$rhs> for $lhs {
            #[inline]
            fn partial_cmp(&self, other: &$rhs) -> Option<Ordering> {
                let other: &[u8] = other.as_ref();
                PartialOrd::partial_cmp(self.as_bytes(), other)
            }
        }

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<'a> PartialOrd<$lhs> for $rhs {
            #[inline]
            fn partial_cmp(&self, other: &$lhs) -> Option<Ordering> {
                let this: &[u8] = self.as_ref();
                PartialOrd::partial_cmp(this, other.as_bytes())
            }
        }
    };
}

#[doc(hidden)]
#[unstable(feature = "bstr_internals", issue = "none")]
pub use impl_partial_eq_ord;

#[doc(hidden)]
#[macro_export]
#[unstable(feature = "bstr_internals", issue = "none")]
macro_rules! impl_partial_eq_n {
    ($lhs:ty, $rhs:ty) => {
        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<const N: usize> PartialEq<$rhs> for $lhs {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                let other: &[u8] = other.as_ref();
                PartialEq::eq(self.as_bytes(), other)
            }
        }

        #[allow(unused_lifetimes)]
        #[unstable(feature = "bstr", issue = "134915")]
        impl<const N: usize> PartialEq<$lhs> for $rhs {
            #[inline]
            fn eq(&self, other: &$lhs) -> bool {
                let this: &[u8] = self.as_ref();
                PartialEq::eq(this, other.as_bytes())
            }
        }
    };
}

#[doc(hidden)]
#[unstable(feature = "bstr_internals", issue = "none")]
pub use impl_partial_eq_n;

// PartialOrd with `[u8]` omitted to avoid inference failures
impl_partial_eq!(ByteStr, [u8]);
// PartialOrd with `&[u8]` omitted to avoid inference failures
impl_partial_eq!(ByteStr, &[u8]);
// PartialOrd with `str` omitted to avoid inference failures
impl_partial_eq!(ByteStr, str);
// PartialOrd with `&str` omitted to avoid inference failures
impl_partial_eq!(ByteStr, &str);
// PartialOrd with `[u8; N]` omitted to avoid inference failures
impl_partial_eq_n!(ByteStr, [u8; N]);
// PartialOrd with `[u8; N]` omitted to avoid inference failures
impl_partial_eq_n!(ByteStr, &[u8; N]);

#[unstable(feature = "bstr", issue = "134915")]
impl Index<usize> for ByteStr {
    type Output = u8;

    #[inline]
    fn index(&self, idx: usize) -> &u8 {
        &self.0[idx]
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeFull> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, _: RangeFull) -> &ByteStr {
        self
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<Range<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: Range<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeInclusive<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeInclusive<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeFrom<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeFrom<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeTo<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeTo<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl Index<RangeToInclusive<usize>> for ByteStr {
    type Output = ByteStr;

    #[inline]
    fn index(&self, r: RangeToInclusive<usize>) -> &ByteStr {
        ByteStr::from_bytes(&self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<usize> for ByteStr {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut u8 {
        &mut self.0[idx]
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeFull> for ByteStr {
    #[inline]
    fn index_mut(&mut self, _: RangeFull) -> &mut ByteStr {
        self
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<Range<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: Range<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeInclusive<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: RangeInclusive<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeFrom<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: RangeFrom<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeTo<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: RangeTo<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl IndexMut<RangeToInclusive<usize>> for ByteStr {
    #[inline]
    fn index_mut(&mut self, r: RangeToInclusive<usize>) -> &mut ByteStr {
        ByteStr::from_bytes_mut(&mut self.0[r])
    }
}
