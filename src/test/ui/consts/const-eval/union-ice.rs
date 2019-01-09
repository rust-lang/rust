#![feature(const_fn)]

type Field1 = i32;
type Field3 = i64;

union DummyUnion {
    field1: Field1,
    field3: Field3,
}

const UNION: DummyUnion = DummyUnion { field1: 1065353216 };

const FIELD3: Field3 = unsafe { UNION.field3 }; //~ ERROR it is undefined behavior to use this value

const FIELD_PATH: Struct = Struct { //~ ERROR it is undefined behavior to use this value
    a: 42,
    b: unsafe { UNION.field3 },
};

struct Struct {
    a: u8,
    b: Field3,
}

const FIELD_PATH2: Struct2 = Struct2 { //~ ERROR it is undefined behavior to use this value
    b: [
        21,
        unsafe { UNION.field3 },
        23,
        24,
    ],
    a: 42,
};

struct Struct2 {
    b: [Field3; 4],
    a: u8,
}

fn main() {
}
