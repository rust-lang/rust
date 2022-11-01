// run-pass
// aux-build:macro-dump-debug.rs

extern crate macro_dump_debug;
use macro_dump_debug::dump_debug;

dump_debug! {
    ident   // ident
    r#ident // raw ident
    ,       // alone punct
    ==>     // joint punct
    ()      // empty group
    [_]     // nonempty group

    // unsuffixed literals
    0
    1.0
    "S"
    b"B"
    r"R"
    r##"R"##
    br"BR"
    br##"BR"##
    'C'
    b'B'
}

fn main() {}
