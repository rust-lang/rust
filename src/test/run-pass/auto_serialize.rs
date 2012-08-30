use std;

// These tests used to be separate files, but I wanted to refactor all
// the common code.

import cmp::Eq;
import std::ebml;
import io::Writer;
import std::serialization::{serialize_uint, deserialize_uint};

fn test_ser_and_deser<A:Eq>(a1: A,
                            expected: ~str,
                            ebml_ser_fn: fn(ebml::Writer, A),
                            ebml_deser_fn: fn(ebml::EbmlDeserializer) -> A,
                            io_ser_fn: fn(io::Writer, A)) {

    // check the pretty printer:
    io_ser_fn(io::stdout(), a1);
    let s = io::with_str_writer(|w| io_ser_fn(w, a1) );
    debug!("s == %?", s);
    assert s == expected;

    // check the EBML serializer:
    let buf = io::mem_buffer();
    let w = ebml::Writer(buf as io::Writer);
    ebml_ser_fn(w, a1);
    let d = ebml::Doc(@io::mem_buffer_buf(buf));
    let a2 = ebml_deser_fn(ebml::ebml_deserializer(d));
    io::print(~"\na1 = ");
    io_ser_fn(io::stdout(), a1);
    io::print(~"\na2 = ");
    io_ser_fn(io::stdout(), a2);
    io::print(~"\n");
    assert a1 == a2;

}

#[auto_serialize]
enum expr {
    val(uint),
    plus(@expr, @expr),
    minus(@expr, @expr)
}

impl an_enum : cmp::Eq {
    pure fn eq(&&other: an_enum) -> bool {
        self.v == other.v
    }
}

impl point : cmp::Eq {
    pure fn eq(&&other: point) -> bool {
        self.x == other.x &&
            self.y == other.y
    }
}

impl<T:cmp::Eq> quark<T> : cmp::Eq {
    pure fn eq(&&other: quark<T>) -> bool {
        match self {
          top(ref q) => match other {
            top(ref r) => q == r,
            bottom(_) => false
          },
          bottom(ref q) => match other {
            top(_) => false,
            bottom(ref r) => q == r
          }
        }
    }
}


impl c_like : cmp::Eq {
    pure fn eq(&&other: c_like) -> bool {
        self as int == other as int
    }
}

impl expr : cmp::Eq {
    pure fn eq(&&other: expr) -> bool {
        match self {
            val(e0a) => {
                match other {
                    val(e0b) => e0a == e0b,
                    _ => false
                }
            }
            plus(e0a, e1a) => {
                match other {
                    plus(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            minus(e0a, e1a) => {
                match other {
                    minus(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
        }
    }
}

#[auto_serialize]
type spanned<T> = {lo: uint, hi: uint, node: T};

impl<T:cmp::Eq> spanned<T> : cmp::Eq {
    pure fn eq(&&other: spanned<T>) -> bool {
        self.lo == other.lo && self.hi == other.hi && self.node.eq(other.node)
    }
}

#[auto_serialize]
type spanned_uint = spanned<uint>;

#[auto_serialize]
type some_rec = {v: uint_vec};

#[auto_serialize]
enum an_enum = some_rec;

#[auto_serialize]
type uint_vec = ~[uint];

#[auto_serialize]
type point = {x: uint, y: uint};

#[auto_serialize]
enum quark<T> {
    top(T),
    bottom(T)
}

#[auto_serialize]
type uint_quark = quark<uint>;

#[auto_serialize]
enum c_like { a, b, c }

fn main() {

    test_ser_and_deser(plus(@minus(@val(3u), @val(10u)),
                            @plus(@val(22u), @val(5u))),
                       ~"plus(@minus(@val(3u), @val(10u)), \
                        @plus(@val(22u), @val(5u)))",
                       serialize_expr,
                       deserialize_expr,
                       serialize_expr);

    test_ser_and_deser({lo: 0u, hi: 5u, node: 22u},
                       ~"{lo: 0u, hi: 5u, node: 22u}",
                       serialize_spanned_uint,
                       deserialize_spanned_uint,
                       serialize_spanned_uint);

    test_ser_and_deser(an_enum({v: ~[1u, 2u, 3u]}),
                       ~"an_enum({v: [1u, 2u, 3u]})",
                       serialize_an_enum,
                       deserialize_an_enum,
                       serialize_an_enum);

    test_ser_and_deser({x: 3u, y: 5u},
                       ~"{x: 3u, y: 5u}",
                       serialize_point,
                       deserialize_point,
                       serialize_point);

    test_ser_and_deser(~[1u, 2u, 3u],
                       ~"[1u, 2u, 3u]",
                       serialize_uint_vec,
                       deserialize_uint_vec,
                       serialize_uint_vec);

    test_ser_and_deser(top(22u),
                       ~"top(22u)",
                       serialize_uint_quark,
                       deserialize_uint_quark,
                       serialize_uint_quark);

    test_ser_and_deser(bottom(222u),
                       ~"bottom(222u)",
                       serialize_uint_quark,
                       deserialize_uint_quark,
                       serialize_uint_quark);

    test_ser_and_deser(a,
                       ~"a",
                       serialize_c_like,
                       deserialize_c_like,
                       serialize_c_like);

    test_ser_and_deser(b,
                       ~"b",
                       serialize_c_like,
                       deserialize_c_like,
                       serialize_c_like);
}
