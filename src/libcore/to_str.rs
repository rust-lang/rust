// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use kinds::Copy;
use str;
use vec;

pub trait ToStr { pub pure fn to_str() -> ~str; }

impl int: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::int::str(self) }
}
impl i8: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::i8::str(self) }
}
impl i16: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::i16::str(self) }
}
impl i32: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::i32::str(self) }
}
impl i64: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::i64::str(self) }
}
impl uint: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::uint::str(self) }
}
impl u8: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::u8::str(self) }
}
impl u16: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::u16::str(self) }
}
impl u32: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::u32::str(self) }
}
impl u64: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::u64::str(self) }
}
impl float: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::float::to_str(self, 4u) }
}
impl f32: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::float::to_str(self as float, 4u) }
}
impl f64: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::float::to_str(self as float, 4u) }
}
impl bool: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::bool::to_str(self) }
}
impl (): ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ~"()" }
}
impl ~str: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { copy self }
}
impl &str: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::str::from_slice(self) }
}
impl @str: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ::str::from_slice(self) }
}

impl<A: ToStr Copy, B: ToStr Copy> (A, B): ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str {
        let (a, b) = self;
        ~"(" + a.to_str() + ~", " + b.to_str() + ~")"
    }
}
impl<A: ToStr Copy, B: ToStr Copy, C: ToStr Copy> (A, B, C): ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str {
        let (a, b, c) = self;
        ~"(" + a.to_str() + ~", " + b.to_str() + ~", " + c.to_str() + ~")"
    }
}

impl<A: ToStr> ~[A]: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str {
        unsafe {
            // Bleh -- not really unsafe
            // push_str and push_char
            let mut acc = ~"[", first = true;
            for vec::each(self) |elt| {
                unsafe {
                    if first { first = false; }
                    else { str::push_str(&mut acc, ~", "); }
                    str::push_str(&mut acc, elt.to_str());
                }
            }
            str::push_char(&mut acc, ']');
            move acc
        }
    }
}

impl<A: ToStr> @A: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ~"@" + (*self).to_str() }
}
impl<A: ToStr> ~A: ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { ~"~" + (*self).to_str() }
}

#[cfg(test)]
#[allow(non_implicitly_copyable_typarams)]
mod tests {
    #[test]
    fn test_simple_types() {
        assert 1.to_str() == ~"1";
        assert (-1).to_str() == ~"-1";
        assert 200u.to_str() == ~"200";
        assert 2u8.to_str() == ~"2";
        assert true.to_str() == ~"true";
        assert false.to_str() == ~"false";
        assert ().to_str() == ~"()";
        assert (~"hi").to_str() == ~"hi";
        assert (@"hi").to_str() == ~"hi";
    }

    #[test]
    fn test_tuple_types() {
        assert (1, 2).to_str() == ~"(1, 2)";
        assert (~"a", ~"b", false).to_str() == ~"(a, b, false)";
        assert ((), ((), 100)).to_str() == ~"((), ((), 100))";
    }

    #[test]
    #[ignore]
    fn test_vectors() {
        let x: ~[int] = ~[];
        assert x.to_str() == ~"~[]";
        assert (~[1]).to_str() == ~"~[1]";
        assert (~[1, 2, 3]).to_str() == ~"~[1, 2, 3]";
        assert (~[~[], ~[1], ~[1, 1]]).to_str() ==
               ~"~[~[], ~[1], ~[1, 1]]";
    }

    #[test]
    fn test_pointer_types() {
        assert (@1).to_str() == ~"@1";
        assert (~(true, false)).to_str() == ~"~(true, false)";
    }
}
