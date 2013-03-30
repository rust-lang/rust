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

use str;

pub trait ToStr {
    fn to_str(&self) -> ~str;
}

impl ToStr for bool {
    #[inline(always)]
    fn to_str(&self) -> ~str { ::bool::to_str(*self) }
}
impl ToStr for () {
    #[inline(always)]
    fn to_str(&self) -> ~str { ~"()" }
}

impl<A:ToStr> ToStr for (A,) {
    #[inline(always)]
    fn to_str(&self) -> ~str {
        match *self {
            (ref a,) => {
                ~"(" + a.to_str() + ~", " + ~")"
            }
        }
    }
}

impl<A:ToStr,B:ToStr> ToStr for (A, B) {
    #[inline(always)]
    fn to_str(&self) -> ~str {
        // FIXME(#4760): this causes an llvm assertion
        //let &(ref a, ref b) = self;
        match *self {
            (ref a, ref b) => {
                ~"(" + a.to_str() + ~", " + b.to_str() + ~")"
            }
        }
    }
}
impl<A:ToStr,B:ToStr,C:ToStr> ToStr for (A, B, C) {
    #[inline(always)]
    fn to_str(&self) -> ~str {
        // FIXME(#4760): this causes an llvm assertion
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
    #[inline(always)]
    fn to_str(&self) -> ~str {
        unsafe {
            // FIXME #4568
            // Bleh -- not really unsafe
            // push_str and push_char
            let mut acc = ~"[", first = true;
            for self.each |elt| {
                unsafe {
                    if first { first = false; }
                    else { str::push_str(&mut acc, ~", "); }
                    str::push_str(&mut acc, elt.to_str());
                }
            }
            str::push_char(&mut acc, ']');
            acc
        }
    }
}

impl<A:ToStr> ToStr for ~[A] {
    #[inline(always)]
    fn to_str(&self) -> ~str {
        unsafe {
            // FIXME #4568
            // Bleh -- not really unsafe
            // push_str and push_char
            let mut acc = ~"[", first = true;
            for self.each |elt| {
                unsafe {
                    if first { first = false; }
                    else { str::push_str(&mut acc, ~", "); }
                    str::push_str(&mut acc, elt.to_str());
                }
            }
            str::push_char(&mut acc, ']');
            acc
        }
    }
}

impl<A:ToStr> ToStr for @[A] {
    #[inline(always)]
    fn to_str(&self) -> ~str {
        unsafe {
            // FIXME #4568
            // Bleh -- not really unsafe
            // push_str and push_char
            let mut acc = ~"[", first = true;
            for self.each |elt| {
                unsafe {
                    if first { first = false; }
                    else { str::push_str(&mut acc, ~", "); }
                    str::push_str(&mut acc, elt.to_str());
                }
            }
            str::push_char(&mut acc, ']');
            acc
        }
    }
}

#[cfg(test)]
#[allow(non_implicitly_copyable_typarams)]
mod tests {
    #[test]
    fn test_simple_types() {
        assert!(1i.to_str() == ~"1");
        assert!((-1i).to_str() == ~"-1");
        assert!(200u.to_str() == ~"200");
        assert!(2u8.to_str() == ~"2");
        assert!(true.to_str() == ~"true");
        assert!(false.to_str() == ~"false");
        assert!(().to_str() == ~"()");
        assert!((~"hi").to_str() == ~"hi");
        assert!((@"hi").to_str() == ~"hi");
    }

    #[test]
    fn test_tuple_types() {
        assert!((1, 2).to_str() == ~"(1, 2)");
        assert!((~"a", ~"b", false).to_str() == ~"(a, b, false)");
        assert!(((), ((), 100)).to_str() == ~"((), ((), 100))");
    }

    #[test]
    fn test_vectors() {
        let x: ~[int] = ~[];
        assert!(x.to_str() == ~"[]");
        assert!((~[1]).to_str() == ~"[1]");
        assert!((~[1, 2, 3]).to_str() == ~"[1, 2, 3]");
        assert!((~[~[], ~[1], ~[1, 1]]).to_str() ==
               ~"[[], [1], [1, 1]]");
    }
}
