extern mod std;

// These tests used to be separate files, but I wanted to refactor all
// the common code.

use cmp::Eq;
use std::ebml2;
use io::Writer;
use std::serialization2::{Serializer, Serializable, deserialize};
use std::prettyprint2;

fn test_ser_and_deser<A:Eq Serializable>(
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

fn main() {
    test_ser_and_deser(Plus(@Minus(@Val(3u), @Val(10u)),
                            @Plus(@Val(22u), @Val(5u))),
                       ~"Plus(@Minus(@Val(3u), @Val(10u)), \
                        @Plus(@Val(22u), @Val(5u)))");
}
