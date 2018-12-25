// run-pass

#![feature(const_fn)]

type Field1 = (i32, u32);
type Field2 = f32;
type Field3 = i64;

union DummyUnion {
    field1: Field1,
    field2: Field2,
    field3: Field3,
}

const FLOAT1_AS_I32: i32 = 1065353216;
const UNION: DummyUnion = DummyUnion { field1: (FLOAT1_AS_I32, 0) };

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
    FIELD3
}

fn main() {
    let foo = FLOAT1_AS_I32;
    assert_eq!(read_field1().0, foo);
    assert_eq!(read_field1().0, FLOAT1_AS_I32);

    let foo = 1.0;
    assert_eq!(read_field2(), foo);
    assert_eq!(read_field2(), 1.0);

    assert_eq!(read_field3(), unsafe { UNION.field3 });
    let foo = unsafe { UNION.field3 };
    assert_eq!(read_field3(), foo);
}
