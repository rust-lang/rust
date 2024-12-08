#![warn(clippy::init_numbered_fields)]

#[derive(Default)]
struct TupleStruct(u32, u32, u8);

// This shouldn't lint because it's in a macro
macro_rules! tuple_struct_init {
    () => {
        TupleStruct { 0: 0, 1: 1, 2: 2 }
    };
}

fn main() {
    let tuple_struct = TupleStruct::default();

    // This should lint
    let _ = TupleStruct {
        0: 1u32,
        1: 42,
        2: 23u8,
    };

    // This should also lint and order the fields correctly
    let _ = TupleStruct {
        0: 1u32,
        2: 2u8,
        1: 3u32,
    };

    // Ok because of default initializer
    let _ = TupleStruct { 0: 42, ..tuple_struct };

    let _ = TupleStruct {
        1: 23,
        ..TupleStruct::default()
    };

    // Ok because it's in macro
    let _ = tuple_struct_init!();

    type Alias = TupleStruct;

    // Aliases can't be tuple constructed #8638
    let _ = Alias { 0: 0, 1: 1, 2: 2 };

    // Issue #12367
    struct TupleStructVec(Vec<usize>);

    let _ = TupleStructVec { 0: vec![0, 1, 2, 3] };

    {
        struct S(i32, i32);
        let mut iter = [1i32, 1i32].into_iter();
        let _ = S {
            1: iter.next().unwrap(),
            0: iter.next().unwrap(),
        };
    }
}
