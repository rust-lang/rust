//@ dont-require-annotations: NOTE
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([[:xdigit:]]{2}\s){4}(__\s){4}\s+│\s+([?|\.]){4}\W{4}" -> "HEX_DUMP"

type Field1 = i32;
type Field2 = f32;
type Field3 = i64;

#[repr(C)]
union DummyUnion {
    field1: Field1,
    field2: Field2,
    field3: Field3,
}

const FLOAT1_AS_I32: i32 = 1065353216;
const UNION: DummyUnion = DummyUnion { field1: FLOAT1_AS_I32 };

const fn read_field1() -> Field1 {
    const FIELD1: Field1 = unsafe { UNION.field1 };
    FIELD1
}

const fn read_field2() -> Field2 {
    const FIELD2: Field2 = unsafe { UNION.field2 };
    FIELD2
}

const fn read_field3() -> Field3 {
    const FIELD3: Field3 = unsafe { UNION.field3 };
    //~^ ERROR uninitialized
    FIELD3
}

fn main() {
    assert_eq!(read_field1(), FLOAT1_AS_I32);
    assert_eq!(read_field2(), 1.0);
    assert_eq!(read_field3(), unsafe { UNION.field3 });
}
