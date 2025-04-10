//@ only-x86_64

type Field1 = i32;
type Field3 = i64;

#[repr(C)]
union DummyUnion {
    field1: Field1,
    field3: Field3,
}

const UNION: DummyUnion = DummyUnion { field1: 1065353216 };

const FIELD3: Field3 = unsafe { UNION.field3 };
//~^ ERROR evaluation of constant value failed
//~| NOTE uninitialized

const FIELD_PATH: Struct = Struct {
    a: 42,
    b: unsafe { UNION.field3 },
    //~^ ERROR evaluation of constant value failed
    //~| NOTE uninitialized
};

struct Struct {
    a: u8,
    b: Field3,
}

const FIELD_PATH2: Struct2 = Struct2 {
    b: [
        21,
        unsafe { UNION.field3 },
        //~^ ERROR evaluation of constant value failed
        //~| NOTE uninitialized
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
