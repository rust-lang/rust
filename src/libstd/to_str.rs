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

use str::OwnedStr;
use hashmap::HashMap;
use hashmap::HashSet;
use hash::Hash;
use iterator::IteratorUtil;
use cmp::Eq;
use vec::ImmutableVector;

/// A generic trait for converting a value to a string
pub trait ToStr {
    /// Converts the value of `self` to an owned string
    fn to_str(&self) -> ~str;
}

/// Trait for converting a type to a string, consuming it in the process.
pub trait ToStrConsume {
    /// Cosume and convert to a string.
    fn into_str(self) -> ~str;
}

impl ToStr for () {
    #[inline]
    fn to_str(&self) -> ~str { ~"()" }
}

impl<A:ToStr> ToStr for (A,) {
    #[inline]
    fn to_str(&self) -> ~str {
        match *self {
            (ref a,) => {
                fmt!("(%s,)", (*a).to_str())
            }
        }
    }
}

impl<A:ToStr+Hash+Eq, B:ToStr+Hash+Eq> ToStr for HashMap<A, B> {
    #[inline]
    fn to_str(&self) -> ~str {
        let mut acc = ~"{";
        let mut first = true;
        for self.iter().advance |(key, value)| {
            if first {
                first = false;
            }
            else {
                acc.push_str(", ");
            }
            acc.push_str(key.to_str());
            acc.push_str(": ");
            acc.push_str(value.to_str());
        }
        acc.push_char('}');
        acc
    }
}

impl<A:ToStr+Hash+Eq> ToStr for HashSet<A> {
    #[inline]
    fn to_str(&self) -> ~str {
        let mut acc = ~"{";
        let mut first = true;
        for self.iter().advance |element| {
            if first {
                first = false;
            }
            else {
                acc.push_str(", ");
            }
            acc.push_str(element.to_str());
        }
        acc.push_char('}');
        acc
    }
}

impl<A:ToStr,B:ToStr> ToStr for (A, B) {
    #[inline]
    fn to_str(&self) -> ~str {
        // FIXME(#4653): this causes an llvm assertion
        //let &(ref a, ref b) = self;
        match *self {
            (ref a, ref b) => {
                fmt!("(%s, %s)", (*a).to_str(), (*b).to_str())
            }
        }
    }
}

impl<A:ToStr,B:ToStr,C:ToStr> ToStr for (A, B, C) {
    #[inline]
    fn to_str(&self) -> ~str {
        // FIXME(#4653): this causes an llvm assertion
        //let &(ref a, ref b, ref c) = self;
        match *self {
            (ref a, ref b, ref c) => {
                fmt!("(%s, %s, %s)",
                    (*a).to_str(),
                    (*b).to_str(),
                    (*c).to_str()
                )
            }
        }
    }
}

impl<'self,A:ToStr> ToStr for &'self [A] {
    #[inline]
    fn to_str(&self) -> ~str {
        let mut acc = ~"[";
        let mut first = true;
        for self.iter().advance |elt| {
            if first {
                first = false;
            }
            else {
                acc.push_str(", ");
            }
            acc.push_str(elt.to_str());
        }
        acc.push_char(']');
        acc
    }
}

impl<A:ToStr> ToStr for ~[A] {
    #[inline]
    fn to_str(&self) -> ~str {
        let mut acc = ~"[";
        let mut first = true;
        for self.iter().advance |elt| {
            if first {
                first = false;
            }
            else {
                acc.push_str(", ");
            }
            acc.push_str(elt.to_str());
        }
        acc.push_char(']');
        acc
    }
}

impl<A:ToStr> ToStr for @[A] {
    #[inline]
    fn to_str(&self) -> ~str {
        let mut acc = ~"[";
        let mut first = true;
        for self.iter().advance |elt| {
            if first {
                first = false;
            }
            else {
                acc.push_str(", ");
            }
            acc.push_str(elt.to_str());
        }
        acc.push_char(']');
        acc
    }
}

#[cfg(test)]
mod tests {
    use hashmap::HashMap;
    use hashmap::HashSet;
    use container::{Set, Map, MutableSet, MutableMap};
    #[test]
    fn test_simple_types() {
        assert_eq!(1i.to_str(), ~"1");
        assert_eq!((-1i).to_str(), ~"-1");
        assert_eq!(200u.to_str(), ~"200");
        assert_eq!(2u8.to_str(), ~"2");
        assert_eq!(true.to_str(), ~"true");
        assert_eq!(false.to_str(), ~"false");
        assert_eq!(().to_str(), ~"()");
        assert_eq!((~"hi").to_str(), ~"hi");
        assert_eq!((@"hi").to_str(), ~"hi");
    }

    #[test]
    fn test_tuple_types() {
        assert_eq!((1, 2).to_str(), ~"(1, 2)");
        assert_eq!((~"a", ~"b", false).to_str(), ~"(a, b, false)");
        assert_eq!(((), ((), 100)).to_str(), ~"((), ((), 100))");
    }

    #[test]
    fn test_vectors() {
        let x: ~[int] = ~[];
        assert_eq!(x.to_str(), ~"[]");
        assert_eq!((~[1]).to_str(), ~"[1]");
        assert_eq!((~[1, 2, 3]).to_str(), ~"[1, 2, 3]");
        assert!((~[~[], ~[1], ~[1, 1]]).to_str() ==
               ~"[[], [1], [1, 1]]");
    }

    #[test]
    fn test_hashmap() {
        let mut table: HashMap<int, int> = HashMap::new();
        let empty: HashMap<int, int> = HashMap::new();

        table.insert(3, 4);
        table.insert(1, 2);

        let table_str = table.to_str();

        assert!(table_str == ~"{1: 2, 3: 4}" || table_str == ~"{3: 4, 1: 2}");
        assert_eq!(empty.to_str(), ~"{}");
    }

    #[test]
    fn test_hashset() {
        let mut set: HashSet<int> = HashSet::new();
        let empty_set: HashSet<int> = HashSet::new();

        set.insert(1);
        set.insert(2);

        let set_str = set.to_str();

        assert!(set_str == ~"{1, 2}" || set_str == ~"{2, 1}");
        assert_eq!(empty_set.to_str(), ~"{}");
    }
}
