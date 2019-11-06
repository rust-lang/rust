#![allow(unused_variables)]
#![allow(non_camel_case_types)]
#![deny(dead_code)]

struct Foo {
    x: usize,
    b: bool, //~ ERROR: field is never used
}

fn field_read(f: Foo) -> usize {
    f.x.pow(2)
}

enum XYZ {
    X, //~ ERROR variant is never constructed
    Y { //~ ERROR variant is never constructed
        a: String,
        b: i32,
        c: i32,
    },
    Z
}

enum ABC { //~ ERROR enum is never used
    A,
    B {
        a: String,
        b: i32,
        c: i32,
    },
    C
}

// ensure struct variants get warning for their fields
enum IJK {
    I, //~ ERROR variant is never constructed
    J {
        a: String,
        b: i32, //~ ERROR field is never used
        c: i32, //~ ERROR field is never used
    },
    K //~ ERROR variant is never constructed

}

fn struct_variant_partial_use(b: IJK) -> String {
    match b {
        IJK::J { a, b: _, .. } => a,
        _ => "".to_string()
    }
}

fn field_match_in_patterns(b: XYZ) -> String {
    match b {
        XYZ::Y { a, b: _, .. } => a,
        _ => "".to_string()
    }
}

struct Bar {
    x: usize, //~ ERROR: field is never used
    b: bool,
    c: bool, //~ ERROR: field is never used
    _guard: ()
}

#[repr(C)]
struct Baz {
    x: u32,
}

fn field_match_in_let(f: Bar) -> bool {
    let Bar { b, c: _, .. } = f;
    b
}

fn main() {
    field_read(Foo { x: 1, b: false });
    field_match_in_patterns(XYZ::Z);
    struct_variant_partial_use(IJK::J { a: "".into(), b: 1, c: -1 });
    field_match_in_let(Bar { x: 42, b: true, c: false, _guard: () });
    let _ = Baz { x: 0 };
}
