// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! `SendStr` definition and trait implementations

use clone::{Clone, DeepClone};
use cmp::{Eq, TotalEq, Ord, TotalOrd, Equiv};
use cmp::Ordering;
use container::Container;
use default::Default;
use str::{Str, StrSlice};
use to_str::ToStr;
use to_bytes::{IterBytes, Cb};

/// A SendStr is a string that can hold either a ~str or a &'static str.
/// This can be useful as an optimization when an allocation is sometimes
/// needed but the common case is statically known.
#[allow(missing_doc)]
pub enum SendStr {
    SendStrOwned(~str),
    SendStrStatic(&'static str)
}

impl SendStr {
    /// Returns `true` if this `SendStr` wraps an owned string
    #[inline]
    pub fn is_owned(&self) -> bool {
        match *self {
            SendStrOwned(_) => true,
            SendStrStatic(_) => false
        }
    }

    /// Returns `true` if this `SendStr` wraps an static string
    #[inline]
    pub fn is_static(&self) -> bool {
        match *self {
            SendStrOwned(_) => false,
            SendStrStatic(_) => true
        }
    }
}

/// Trait for moving into an `SendStr`
pub trait IntoSendStr {
    /// Moves self into an `SendStr`
    fn into_send_str(self) -> SendStr;
}

impl IntoSendStr for ~str {
    #[inline]
    fn into_send_str(self) -> SendStr { SendStrOwned(self) }
}

impl IntoSendStr for &'static str {
    #[inline]
    fn into_send_str(self) -> SendStr { SendStrStatic(self) }
}

impl IntoSendStr for SendStr {
    #[inline]
    fn into_send_str(self) -> SendStr { self }
}

/*
Section: String trait impls.
`SendStr` should behave like a normal string, so we don't derive.
*/

impl ToStr for SendStr {
    #[inline]
    fn to_str(&self) -> ~str { self.as_slice().to_owned() }
}

impl Eq for SendStr {
    #[inline]
    fn eq(&self, other: &SendStr) -> bool {
        self.as_slice().equals(&other.as_slice())
    }
}

impl TotalEq for SendStr {
    #[inline]
    fn equals(&self, other: &SendStr) -> bool {
        self.as_slice().equals(&other.as_slice())
    }
}

impl Ord for SendStr {
    #[inline]
    fn lt(&self, other: &SendStr) -> bool {
        self.as_slice().lt(&other.as_slice())
    }
}

impl TotalOrd for SendStr {
    #[inline]
    fn cmp(&self, other: &SendStr) -> Ordering {
        self.as_slice().cmp(&other.as_slice())
    }
}

impl<'a, S: Str> Equiv<S> for SendStr {
    #[inline]
    fn equiv(&self, other: &S) -> bool {
        self.as_slice().equals(&other.as_slice())
    }
}

impl Str for SendStr {
    #[inline]
    fn as_slice<'r>(&'r self) -> &'r str {
        match *self {
            SendStrOwned(ref s) => s.as_slice(),
            // XXX: Borrowchecker doesn't recognize lifetime as static unless prompted
            // SendStrStatic(s) => s.as_slice()
            SendStrStatic(s)    => {let tmp: &'static str = s; tmp}
        }
    }

    #[inline]
    fn into_owned(self) -> ~str {
        match self {
            SendStrOwned(s)  => s,
            SendStrStatic(s) => s.to_owned()
        }
    }
}

impl Container for SendStr {
    #[inline]
    fn len(&self) -> uint { self.as_slice().len() }
}

impl Clone for SendStr {
    #[inline]
    fn clone(&self) -> SendStr {
        match *self {
            SendStrOwned(ref s) => SendStrOwned(s.to_owned()),
            SendStrStatic(s)    => SendStrStatic(s)
        }
    }
}

impl DeepClone for SendStr {
    #[inline]
    fn deep_clone(&self) -> SendStr {
        match *self {
            SendStrOwned(ref s) => SendStrOwned(s.to_owned()),
            SendStrStatic(s)    => SendStrStatic(s)
        }
    }
}

impl Default for SendStr {
    #[inline]
    fn default() -> SendStr { SendStrStatic("") }
}

impl IterBytes for SendStr {
    #[inline]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        match *self {
            SendStrOwned(ref s) => s.iter_bytes(lsb0, f),
            SendStrStatic(s)    => s.iter_bytes(lsb0, f)
        }
    }
}

#[cfg(test)]
mod tests {
    use clone::{Clone, DeepClone};
    use cmp::{TotalEq, Ord, TotalOrd, Equiv};
    use cmp::Equal;
    use container::Container;
    use default::Default;
    use send_str::{SendStrOwned, SendStrStatic};
    use str::Str;
    use to_str::ToStr;

    #[test]
    fn test_send_str_traits() {
        let s = SendStrStatic("abcde");
        assert_eq!(s.len(), 5);
        assert_eq!(s.as_slice(), "abcde");
        assert_eq!(s.to_str(), ~"abcde");
        assert!(s.equiv(&@"abcde"));
        assert!(s.lt(&SendStrOwned(~"bcdef")));
        assert_eq!(SendStrStatic(""), Default::default());

        let o = SendStrOwned(~"abcde");
        assert_eq!(o.len(), 5);
        assert_eq!(o.as_slice(), "abcde");
        assert_eq!(o.to_str(), ~"abcde");
        assert!(o.equiv(&@"abcde"));
        assert!(o.lt(&SendStrStatic("bcdef")));
        assert_eq!(SendStrOwned(~""), Default::default());

        assert_eq!(s.cmp(&o), Equal);
        assert!(s.equals(&o));
        assert!(s.equiv(&o));

        assert_eq!(o.cmp(&s), Equal);
        assert!(o.equals(&s));
        assert!(o.equiv(&s));
    }

    #[test]
    fn test_send_str_methods() {
        let s = SendStrStatic("abcde");
        assert!(s.is_static());
        assert!(!s.is_owned());

        let o = SendStrOwned(~"abcde");
        assert!(!o.is_static());
        assert!(o.is_owned());
    }

    #[test]
    fn test_send_str_clone() {
        assert_eq!(SendStrOwned(~"abcde"), SendStrStatic("abcde").clone());
        assert_eq!(SendStrOwned(~"abcde"), SendStrStatic("abcde").deep_clone());

        assert_eq!(SendStrOwned(~"abcde"), SendStrOwned(~"abcde").clone());
        assert_eq!(SendStrOwned(~"abcde"), SendStrOwned(~"abcde").deep_clone());

        assert_eq!(SendStrStatic("abcde"), SendStrStatic("abcde").clone());
        assert_eq!(SendStrStatic("abcde"), SendStrStatic("abcde").deep_clone());

        assert_eq!(SendStrStatic("abcde"), SendStrOwned(~"abcde").clone());
        assert_eq!(SendStrStatic("abcde"), SendStrOwned(~"abcde").deep_clone());
    }

    #[test]
    fn test_send_str_into_owned() {
        assert_eq!(SendStrStatic("abcde").into_owned(), ~"abcde");
        assert_eq!(SendStrOwned(~"abcde").into_owned(), ~"abcde");
    }

    #[test]
    fn test_into_send_str() {
        assert_eq!("abcde".into_send_str(), SendStrStatic("abcde"));
        assert_eq!((~"abcde").into_send_str(), SendStrStatic("abcde"));
        assert_eq!("abcde".into_send_str(), SendStrOwned(~"abcde"));
        assert_eq!((~"abcde").into_send_str(), SendStrOwned(~"abcde"));
    }
}
