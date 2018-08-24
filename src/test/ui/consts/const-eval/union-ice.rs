#![feature(const_fn)]

type Field1 = i32;
type Field3 = i64;

union DummyUnion {
    field1: Field1,
    field3: Field3,
}

const UNION: DummyUnion = DummyUnion { field1: 1065353216 };

const FIELD3: Field3 = unsafe { UNION.field3 }; //~ ERROR this constant cannot be used

const FIELD_PATH: Struct = Struct { //~ ERROR this constant cannot be used
    a: 42,
    b: unsafe { UNION.field3 },
};

struct Struct {
    a: u8,
    b: Field3,
}

const FIELD_PATH2: Struct2 = Struct2 { //~ ERROR this constant likely exhibits undefined behavior
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
