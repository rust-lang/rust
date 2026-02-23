//@ compile-flags:-Zunstable-options --remap-path-prefix={{src-base}}=

pub struct MyStruct {
    field: u32,
}

impl MyStruct {
    pub fn new() -> MyStruct {
        MyStruct { field: 3 }
    }
}
