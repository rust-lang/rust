#![feature(rustc_attrs)]

extern //~ WARN missing_abi
    "C"suffix //~ ERROR suffixes on string literals are invalid
    fn foo() {}

extern //~ WARN missing_abi
    "C"suffix //~ ERROR suffixes on string literals are invalid
{}

fn main() {
    ""suffix; //~ ERROR suffixes on string literals are invalid
    b""suffix; //~ ERROR suffixes on byte string literals are invalid
    r#""#suffix; //~ ERROR suffixes on string literals are invalid
    br#""#suffix; //~ ERROR suffixes on byte string literals are invalid
    'a'suffix; //~ ERROR suffixes on char literals are invalid
    b'a'suffix; //~ ERROR suffixes on byte literals are invalid

    1234u1024; //~ ERROR invalid width `1024` for integer literal
    1234i1024; //~ ERROR invalid width `1024` for integer literal
    1234f1024; //~ ERROR invalid width `1024` for float literal
    1234.5f1024; //~ ERROR invalid width `1024` for float literal

    1234suffix; //~ ERROR invalid suffix `suffix` for number literal
    0b101suffix; //~ ERROR invalid suffix `suffix` for number literal
    1.0suffix; //~ ERROR invalid suffix `suffix` for float literal
    1.0e10suffix; //~ ERROR invalid suffix `suffix` for float literal
}

#[rustc_dummy = "string"suffix]
//~^ ERROR suffixes on string literals are invalid
fn f() {}

#[must_use = "string"suffix]
//~^ ERROR suffixes on string literals are invalid
fn g() {}

#[link(name = "string"suffix)]
//~^ ERROR suffixes on string literals are invalid
extern "C" {}

#[rustc_layout_scalar_valid_range_start(0suffix)]
//~^ ERROR invalid suffix `suffix` for number literal
struct S;
