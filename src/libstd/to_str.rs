// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The `ToStr` trait for converting to strings

*/

use fmt;
use strbuf::StrBuf;

/// A generic trait for converting a value to a string
pub trait ToStr {
    /// Converts the value of `self` to an owned string
    fn to_str(&self) -> StrBuf;
}

/// Trait for converting a type to a string, consuming it in the process.
pub trait IntoStr {
    /// Consume and convert to a string.
    fn into_str(self) -> StrBuf;
}

impl<T: fmt::Show> ToStr for T {
    fn to_str(&self) -> StrBuf {
        format_strbuf!("{}", *self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use str::StrAllocating;

    #[test]
    fn test_simple_types() {
        assert_eq!(1i.to_str(), "1".to_strbuf());
        assert_eq!((-1i).to_str(), "-1".to_strbuf());
        assert_eq!(200u.to_str(), "200".to_strbuf());
        assert_eq!(2u8.to_str(), "2".to_strbuf());
        assert_eq!(true.to_str(), "true".to_strbuf());
        assert_eq!(false.to_str(), "false".to_strbuf());
        assert_eq!(().to_str(), "()".to_strbuf());
        assert_eq!(("hi".to_strbuf()).to_str(), "hi".to_strbuf());
    }

    #[test]
    fn test_vectors() {
        let x: ~[int] = box [];
        assert_eq!(x.to_str(), "[]".to_strbuf());
        assert_eq!((box [1]).to_str(), "[1]".to_strbuf());
        assert_eq!((box [1, 2, 3]).to_str(), "[1, 2, 3]".to_strbuf());
        assert!((box [box [], box [1], box [1, 1]]).to_str() ==
               "[[], [1], [1, 1]]".to_strbuf());
    }
}
