// Test invoked `&self` methods on owned objects where the values
// closed over do not contain managed values, and thus the boxes do
// not have headers.

#![feature(box_syntax)]


trait FooTrait {
    fn foo(&self) -> usize;
}

struct BarStruct {
    x: usize
}

impl FooTrait for BarStruct {
    fn foo(&self) -> usize {
        self.x
    }
}

pub fn main() {
    let foos: Vec<Box<dyn FooTrait>> = vec![
        box BarStruct{ x: 0 } as Box<dyn FooTrait>,
        box BarStruct{ x: 1 } as Box<dyn FooTrait>,
        box BarStruct{ x: 2 } as Box<dyn FooTrait>
    ];

    for i in 0..foos.len() {
        assert_eq!(i, foos[i].foo());
    }
}
