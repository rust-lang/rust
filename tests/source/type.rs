fn types() {
    let x: [ Vec   < _ > ] = [];
    let y:  * mut [ SomeType ; konst_funk() ] = expr();
    let z: (/*#digits*/ usize, /*exp*/ i16) = funk();
    let z: ( usize  /*#digits*/ , i16 /*exp*/ ) = funk();
}

struct F {
    f: extern "C" fn(x: u8, ... /* comment */),
    g: extern "C" fn(x: u8,/* comment */ ...),
    h: extern "C" fn(x: u8, ... ),
    i: extern "C" fn(x: u8, /* comment 4*/ y: String, // comment 3
                     z: Foo, /* comment */ .../* comment 2*/ ),
}
