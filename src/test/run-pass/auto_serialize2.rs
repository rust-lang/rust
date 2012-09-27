extern mod std;

// These tests used to be separate files, but I wanted to refactor all
// the common code.

use cmp::Eq;
use std::ebml2;
use io::Writer;
use std::serialization2::{Serializable, Deserializable, deserialize};
use std::prettyprint2;

fn test_ser_and_deser<A:Eq Serializable Deserializable>(
    a1: A,
    expected: ~str
) {
    // check the pretty printer:
    let s = do io::with_str_writer |w| {
        a1.serialize(&prettyprint2::Serializer(w))
    };
    debug!("s == %?", s);
    assert s == expected;

    // check the EBML serializer:
    let bytes = do io::with_bytes_writer |wr| {
        let ebml_w = &ebml2::Serializer(wr);
        a1.serialize(ebml_w)
    };
    let d = ebml2::Doc(@bytes);
    let a2: A = deserialize(&ebml2::Deserializer(d));
    assert a1 == a2;
}

#[auto_serialize2]
enum Expr {
    Val(uint),
    Plus(@Expr, @Expr),
    Minus(@Expr, @Expr)
}

impl Expr : cmp::Eq {
    pure fn eq(other: &Expr) -> bool {
        match self {
            Val(e0a) => {
                match *other {
                    Val(e0b) => e0a == e0b,
                    _ => false
                }
            }
            Plus(e0a, e1a) => {
                match *other {
                    Plus(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            Minus(e0a, e1a) => {
                match *other {
                    Minus(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &Expr) -> bool { !self.eq(other) }
}

impl AnEnum : cmp::Eq {
    pure fn eq(other: &AnEnum) -> bool {
        self.v == other.v
    }
    pure fn ne(other: &AnEnum) -> bool { !self.eq(other) }
}

impl Point : cmp::Eq {
    pure fn eq(other: &Point) -> bool {
        self.x == other.x && self.y == other.y
    }
    pure fn ne(other: &Point) -> bool { !self.eq(other) }
}

impl<T:cmp::Eq> Quark<T> : cmp::Eq {
    pure fn eq(other: &Quark<T>) -> bool {
        match self {
            Top(ref q) => {
                match *other {
                    Top(ref r) => q == r,
                    Bottom(_) => false
                }
            },
            Bottom(ref q) => {
                match *other {
                    Top(_) => false,
                    Bottom(ref r) => q == r
                }
            },
        }
    }
    pure fn ne(other: &Quark<T>) -> bool { !self.eq(other) }
}

impl CLike : cmp::Eq {
    pure fn eq(other: &CLike) -> bool {
        self as int == *other as int
    }
    pure fn ne(other: &CLike) -> bool { !self.eq(other) }
}

#[auto_serialize2]
type Spanned<T> = {lo: uint, hi: uint, node: T};

impl<T:cmp::Eq> Spanned<T> : cmp::Eq {
    pure fn eq(other: &Spanned<T>) -> bool {
        self.lo == other.lo && self.hi == other.hi && self.node == other.node
    }
    pure fn ne(other: &Spanned<T>) -> bool { !self.eq(other) }
}

#[auto_serialize2]
type SomeRec = {v: ~[uint]};

#[auto_serialize2]
enum AnEnum = SomeRec;

#[auto_serialize2]
type Point = {x: uint, y: uint};

#[auto_serialize2]
enum Quark<T> {
    Top(T),
    Bottom(T)
}

#[auto_serialize2]
enum CLike { A, B, C }

fn main() {
    test_ser_and_deser(Plus(@Minus(@Val(3u), @Val(10u)),
                            @Plus(@Val(22u), @Val(5u))),
                       ~"Plus(@Minus(@Val(3u), @Val(10u)), \
                        @Plus(@Val(22u), @Val(5u)))");

    test_ser_and_deser({lo: 0u, hi: 5u, node: 22u},
                       ~"{lo: 0u, hi: 5u, node: 22u}");

    test_ser_and_deser(AnEnum({v: ~[1u, 2u, 3u]}),
                       ~"AnEnum({v: ~[1u, 2u, 3u]})");

    test_ser_and_deser({x: 3u, y: 5u}, ~"{x: 3u, y: 5u}");

    test_ser_and_deser(@[1u, 2u, 3u], ~"@[1u, 2u, 3u]");

    test_ser_and_deser(Top(22u), ~"Top(22u)");
    test_ser_and_deser(Bottom(222u), ~"Bottom(222u)");

    test_ser_and_deser(A, ~"A");
    test_ser_and_deser(B, ~"B");
}
