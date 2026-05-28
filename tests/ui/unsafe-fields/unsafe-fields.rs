//@ compile-flags: --crate-type=lib
#![allow(incomplete_features)]
#![feature(unsafe_fields)]

struct WithUnsafeField {
    unsafe unsafe_field: u32,
    safe_field: u32,
}

enum A {
    WithUnsafeField { unsafe unsafe_field: u32, safe_field: u32 },
}

fn f(a: A) {
    let A::WithUnsafeField { unsafe_field, safe_field } = a;
    //~^ ERROR
}

struct WithInvalidUnsafeField {
    unsafe unsafe_noncopy_field: Vec<u32>,
}

struct WithManuallyDropUnsafeField {
    unsafe unsafe_noncopy_field: std::mem::ManuallyDrop<Vec<u32>>,
}

union WithUnsafeFieldUnion {
    unsafe unsafe_field: u32,
    safe_field: u32,
}

impl WithUnsafeField {
    fn new() -> WithUnsafeField {
        unsafe {
            WithUnsafeField {
                unsafe_field: 0,
                safe_field: 0,
            }
        }
    }

    fn new_without_unsafe() -> WithUnsafeField {
        WithUnsafeField { //~ ERROR
            unsafe_field: 0,
            safe_field: 0,
        }
    }

    fn operate_on_safe_field(&mut self) {
        self.safe_field = 2;
        &self.safe_field;
        self.safe_field;
    }

    fn set_unsafe_field(&mut self) {
        unsafe {
            self.unsafe_field = 2;
        }
    }

    fn read_unsafe_field(&self) -> u32 {
        unsafe {
            self.unsafe_field
        }
    }

    fn ref_unsafe_field(&self) -> &u32 {
        unsafe {
            &self.unsafe_field
        }
    }

    fn destructure(&self) {
        unsafe {
            let Self { safe_field, unsafe_field } = self;
        }
    }

    fn set_unsafe_field_without_unsafe(&mut self) {
        self.unsafe_field = 2;
        //~^ ERROR
    }

    fn read_unsafe_field_without_unsafe(&self) -> u32 {
        self.unsafe_field
        //~^ ERROR
    }

    fn ref_unsafe_field_without_unsafe(&self) -> &u32 {
        &self.unsafe_field
        //~^ ERROR
    }

    fn destructure_without_unsafe(&self) {
        let Self { safe_field, unsafe_field } = self;
        //~^ ERROR

        let WithUnsafeField { safe_field, .. } = self;
    }

    fn offset_of(&self) -> usize {
        std::mem::offset_of!(WithUnsafeField, unsafe_field)
    }

    fn raw_const(&self) -> *const u32 {
        &raw const self.unsafe_field
        //~^ ERROR
    }
}
