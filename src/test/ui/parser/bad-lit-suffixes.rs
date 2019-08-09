extern
    "C"suffix //~ ERROR suffixes on an ABI spec are invalid
    fn foo() {}

extern
    "C"suffix //~ ERROR suffixes on an ABI spec are invalid
{}

fn main() {
    ""suffix; //~ ERROR suffixes on a string literal are invalid
    b""suffix; //~ ERROR suffixes on a byte string literal are invalid
    r#""#suffix; //~ ERROR suffixes on a string literal are invalid
    br#""#suffix; //~ ERROR suffixes on a byte string literal are invalid
    'a'suffix; //~ ERROR suffixes on a char literal are invalid
    b'a'suffix; //~ ERROR suffixes on a byte literal are invalid

    1234u1024; //~ ERROR invalid width `1024` for integer literal
    1234i1024; //~ ERROR invalid width `1024` for integer literal
    1234f1024; //~ ERROR invalid width `1024` for float literal
    1234.5f1024; //~ ERROR invalid width `1024` for float literal

    1234suffix; //~ ERROR invalid suffix `suffix` for integer literal
    0b101suffix; //~ ERROR invalid suffix `suffix` for integer literal
    1.0suffix; //~ ERROR invalid suffix `suffix` for float literal
    1.0e10suffix; //~ ERROR invalid suffix `suffix` for float literal
}
