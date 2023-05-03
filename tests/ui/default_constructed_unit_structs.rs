#![allow(unused)]
#![warn(clippy::default_constructed_unit_structs)]
use std::marker::PhantomData;

#[derive(Default)]
struct UnitStruct;

#[derive(Default)]
struct TupleStruct(usize);

// no lint for derived impl
#[derive(Default)]
struct NormalStruct {
    inner: PhantomData<usize>,
}

struct NonDefaultStruct;

impl NonDefaultStruct {
    fn default() -> Self {
        Self
    }
}

#[derive(Default)]
enum SomeEnum {
    #[default]
    Unit,
    Tuple(UnitStruct),
    Struct {
        inner: usize,
    },
}

impl NormalStruct {
    fn new() -> Self {
        // should lint
        Self {
            inner: PhantomData::default(),
        }
    }
}

#[derive(Default)]
struct GenericStruct<T> {
    t: T,
}

impl<T: Default> GenericStruct<T> {
    fn new() -> Self {
        // should not lint
        Self { t: T::default() }
    }
}

#[derive(Default)]
#[non_exhaustive]
struct NonExhaustiveStruct;

fn main() {
    // should lint
    let _ = PhantomData::<usize>::default();
    let _: PhantomData<i32> = PhantomData::default();
    let _ = UnitStruct::default();

    // should not lint
    let _ = TupleStruct::default();
    let _ = NormalStruct::default();
    let _ = NonExhaustiveStruct::default();
    let _ = SomeEnum::default();
    let _ = NonDefaultStruct::default();
}
