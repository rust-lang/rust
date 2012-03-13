#[link(name="auto_serialize_lib", vers="0.0")];

use std;
import std::ebml;
import io::writer;

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
