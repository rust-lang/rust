use std;

// These tests used to be separate files, but I wanted to refactor all
// the common code.

import std::ebml;
import io::writer;
import std::prettyprint::serializer;
import std::ebml::serializer;
import std::ebml::deserializer;

fn test_ser_and_deser<A>(a1: A,
                         expected: str,
                         ebml_ser_fn: fn(ebml::writer, A),
                         ebml_deser_fn: fn(ebml::ebml_deserializer) -> A,
                         io_ser_fn: fn(io::writer, A)) {

    // check the pretty printer:
    io_ser_fn(io::stdout(), a1);
    let s = io::with_str_writer {|w| io_ser_fn(w, a1) };
    #debug["s == %?", s];
    assert s == expected;

    // check the EBML serializer:
    let buf = io::mem_buffer();
    let w = ebml::writer(buf as io::writer);
    ebml_ser_fn(w, a1);
    let d = ebml::new_doc(@io::mem_buffer_buf(buf));
    let a2 = ebml_deser_fn(ebml::ebml_deserializer(d));
    io::print("\na1 = ");
    io_ser_fn(io::stdout(), a1);
    io::print("\na2 = ");
    io_ser_fn(io::stdout(), a2);
    io::print("\n");
    assert a1 == a2;

}

#[auto_serialize]
enum expr {
    val(uint),
    plus(@expr, @expr),
    minus(@expr, @expr)
}


#[auto_serialize]
type spanned<T> = {lo: uint, hi: uint, node: T};

#[auto_serialize]
type spanned_uint = spanned<uint>;

#[auto_serialize]
type some_rec = {v: uint_vec};

#[auto_serialize]
enum an_enum = some_rec;

#[auto_serialize]
type uint_vec = [uint];

#[auto_serialize]
type point = {x: uint, y: uint};

fn main() {

    test_ser_and_deser(plus(@minus(@val(3u), @val(10u)),
                            @plus(@val(22u), @val(5u))),
                       "plus(@minus(@val(3u), @val(10u)), \
                        @plus(@val(22u), @val(5u)))",
                       expr::serialize(_, _),
                       expr::deserialize(_),
                       expr::serialize(_, _));

    test_ser_and_deser({lo: 0u, hi: 5u, node: 22u},
                       "{lo: 0u, hi: 5u, node: 22u}",
                       spanned_uint::serialize(_, _),
                       spanned_uint::deserialize(_),
                       spanned_uint::serialize(_, _));

    test_ser_and_deser(an_enum({v: [1u, 2u, 3u]}),
                       "an_enum({v: [1u, 2u, 3u]})",
                       an_enum::serialize(_, _),
                       an_enum::deserialize(_),
                       an_enum::serialize(_, _));

    test_ser_and_deser({x: 3u, y: 5u},
                       "{x: 3u, y: 5u}",
                       point::serialize(_, _),
                       point::deserialize(_),
                       point::serialize(_, _));

    test_ser_and_deser([1u, 2u, 3u],
                       "[1u, 2u, 3u]",
                       uint_vec::serialize(_, _),
                       uint_vec::deserialize(_),
                       uint_vec::serialize(_, _));
}