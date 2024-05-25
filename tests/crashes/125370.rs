//@ known-bug: rust-lang/rust#125370

type Field3 = i64;

#[repr(C)]
union DummyUnion {
    field3: Field3,
}

const UNION: DummyUnion = loop {};

const fn read_field2() -> Field2 {
    const FIELD2: Field2 = loop {
        UNION.field3
    };
}
