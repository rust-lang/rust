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

use option::{Some, None};
use str::OwnedStr;
use hashmap::HashMap;
use hashmap::HashSet;
use hash::Hash;
use iterator::Iterator;
use cmp::Eq;
use rt::io::{StringWriter, Decorator};
use rt::io::mem::MemWriter;
use str;
use str::StrSlice;
use vec::{Vector, ImmutableVector};

/// A generic trait for converting a value to a string
pub trait ToStr {
    /// Converts the value of `self` to an owned string
    fn to_str(&self) -> ~str {
        let mut wr = MemWriter::new();
        self.to_str_writer(&mut wr);
        str::from_bytes_owned(wr.inner())
    }

    /// Write a string representation of `self` to `w`
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
        // Passing down `StringWriter` restricts implementors to writing string parts,
        w.write_str(self.to_str());
    }
}

/// Trait for converting a type to a string, consuming it in the process.
pub trait ToStrConsume {
    /// Cosume and convert to a string.
    fn into_str(self) -> ~str;
}

impl ToStr for () {
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) { w.write_str("()"); }
}

/**
* Convert a `bool` to a `str`.
*
* # Examples
*
* ~~~ {.rust}
* rusti> true.to_str()
* "true"
* ~~~
*
* ~~~ {.rust}
* rusti> false.to_str()
* "false"
* ~~~
*/
impl ToStr for bool {
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
        w.write_str(if *self { "true" } else { "false" })
    }
}

impl ToStr for ~str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) { w.write_str(*self); }
}
impl<'self> ToStr for &'self str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) { w.write_str(*self); }
}
impl ToStr for @str {
    #[inline]
    fn to_str(&self) -> ~str { self.to_owned() }
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) { w.write_str(*self); }
}

impl<A: ToStr> ToStr for (A, ) {
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
        w.write_char('(');
        match *self {
            (ref a, ) => a.to_str_writer(w),
        }
        w.write_str(",)");
    }
}

macro_rules! tuple_tostr(
    ($A:ident, $($B:ident),+) => (
        impl<$A: ToStr $(,$B: ToStr)+> ToStr for ($A, $($B),+) {
            #[inline]
            fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
                w.write_char('(');
                match *self {
                    (ref $A, $( ref $B ),+) => {
                        ($A).to_str_writer(w);
                        $(
                            w.write_str(", ");
                            ($B).to_str_writer(w);
                        )*
                    }
                }
                w.write_char(')');
            }
        }
    )
)

tuple_tostr!(A, B)
tuple_tostr!(A, B, C)
tuple_tostr!(A, B, C, D)
tuple_tostr!(A, B, C, D, E)
tuple_tostr!(A, B, C, D, E, F)
tuple_tostr!(A, B, C, D, E, F, G)
tuple_tostr!(A, B, C, D, E, F, G, H)
tuple_tostr!(A, B, C, D, E, F, G, H, I)
tuple_tostr!(A, B, C, D, E, F, G, H, I, J)
tuple_tostr!(A, B, C, D, E, F, G, H, I, J, K)
tuple_tostr!(A, B, C, D, E, F, G, H, I, J, K, L)


impl<'self, A: ToStr> ToStr for &'self [A] {
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
        w.write_char('[');
        let mut first = true;
        foreach element in self.iter() {
            if first {
                first = false;
            }
            else {
                w.write_str(", ");
            }
            element.to_str_writer(w);
        }
        w.write_char(']');
    }
}

impl<A: ToStr> ToStr for ~[A] {
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
        self.as_slice().to_str_writer(w)
    }
}

impl<A:ToStr> ToStr for @[A] {
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
        self.as_slice().to_str_writer(w)
    }
}

impl<A:ToStr+Hash+Eq, B:ToStr> ToStr for HashMap<A, B> {
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
        w.write_char('{');
        let mut first = true;
        foreach (key, value) in self.iter() {
            if first {
                first = false;
            }
            else {
                w.write_str(", ");
            }
            key.to_str_writer(w);
            w.write_str(": ");
            value.to_str_writer(w);
        }
        w.write_char('}');
    }
}

impl<A:ToStr+Hash+Eq> ToStr for HashSet<A> {
    #[inline]
    fn to_str_writer<W: StringWriter>(&self, w: &mut W) {
        w.write_char('{');
        let mut first = true;
        foreach element in self.iter() {
            if first {
                first = false;
            }
            else {
                w.write_str(", ");
            }
            element.to_str_writer(w);
        }
        w.write_char('}');
    }
}

#[cfg(test)]
mod tests {
    extern mod extra;
    use extra::test::BenchHarness;

    use hashmap::HashMap;
    use hashmap::HashSet;
    use uint;
    use container::{Container, MutableSet, MutableMap};
    use str::StrSlice;
    use super::*;

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

    struct StructWithToStrWithoutEqOrHash {
        value: int
    }

    impl ToStr for StructWithToStrWithoutEqOrHash {
        fn to_str(&self) -> ~str {
            fmt!("s%d", self.value)
        }
    }

    #[test]
    fn test_hashmap() {
        let mut table: HashMap<int, StructWithToStrWithoutEqOrHash> = HashMap::new();
        let empty: HashMap<int, StructWithToStrWithoutEqOrHash> = HashMap::new();

        table.insert(3, StructWithToStrWithoutEqOrHash { value: 4 });
        table.insert(1, StructWithToStrWithoutEqOrHash { value: 2 });

        let table_str = table.to_str();

        assert!(table_str == ~"{1: s2, 3: s4}" || table_str == ~"{3: s4, 1: s2}");
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

    #[bench]
    fn bench_hashmap(b: &mut BenchHarness) {
        let mut map = HashMap::new::<uint, &str>();
        let s = "0123456789";
        for uint::range(0, 100) |i| {
            map.insert(i, s.slice(0, i % s.len()));
        }
        do b.iter {
            let s = map.to_str(); s.len();
        }
    }
}
