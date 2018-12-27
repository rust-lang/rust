struct SizedStruct {
    _a: i32,
}

struct UnsizedStruct {
    _a: [i32],
}

struct StructWithVecBox {
    sized_type: Vec<Box<SizedStruct>>,
}

struct StructWithVecBoxButItsUnsized {
    unsized_type: Vec<Box<UnsizedStruct>>,
}

fn main() {}
