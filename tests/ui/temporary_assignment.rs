#![warn(clippy::temporary_assignment)]

use std::ops::{Deref, DerefMut};

struct TupleStruct(i32);

struct Struct {
    field: i32,
}

struct MultiStruct {
    structure: Struct,
}

struct Wrapper<'a> {
    inner: &'a mut Struct,
}

impl<'a> Deref for Wrapper<'a> {
    type Target = Struct;
    fn deref(&self) -> &Struct {
        self.inner
    }
}

impl<'a> DerefMut for Wrapper<'a> {
    fn deref_mut(&mut self) -> &mut Struct {
        self.inner
    }
}

struct ArrayStruct {
    array: [i32; 1],
}

const A: TupleStruct = TupleStruct(1);
const B: Struct = Struct { field: 1 };
const C: MultiStruct = MultiStruct {
    structure: Struct { field: 1 },
};
const D: ArrayStruct = ArrayStruct { array: [1] };

fn main() {
    let mut s = Struct { field: 0 };
    let mut t = (0, 0);

    Struct { field: 0 }.field = 1;
    MultiStruct {
        structure: Struct { field: 0 },
    }
    .structure
    .field = 1;
    ArrayStruct { array: [0] }.array[0] = 1;
    (0, 0).0 = 1;

    // no error
    s.field = 1;
    t.0 = 1;
    Wrapper { inner: &mut s }.field = 1;
    let mut a_mut = TupleStruct(1);
    a_mut.0 = 2;
    let mut b_mut = Struct { field: 1 };
    b_mut.field = 2;
    let mut c_mut = MultiStruct {
        structure: Struct { field: 1 },
    };
    c_mut.structure.field = 2;
    let mut d_mut = ArrayStruct { array: [1] };
    d_mut.array[0] = 2;
}
