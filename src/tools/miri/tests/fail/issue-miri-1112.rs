trait Empty {}

#[repr(transparent)]
pub struct FunnyPointer(dyn Empty);

#[repr(C)]
pub struct Meta {
    drop_fn: fn(&mut ()),
    size: usize,
    align: usize,
}

impl Meta {
    pub fn new() -> Self {
        Meta { drop_fn: |_| {}, size: 0, align: 1 }
    }
}

#[repr(C)]
pub struct FatPointer {
    pub data: *const (),
    pub vtable: *const (),
}

impl FunnyPointer {
    pub unsafe fn from_data_ptr(data: &String, ptr: *const Meta) -> &Self {
        let obj = FatPointer {
            data: data as *const _ as *const (),
            vtable: ptr as *const _ as *const (),
        };
        let obj = std::mem::transmute::<FatPointer, *mut FunnyPointer>(obj); //~ ERROR: expected a vtable pointer
        &*obj
    }
}

fn main() {
    unsafe {
        let meta = Meta::new();
        let hello = "hello".to_string();
        let _raw: &FunnyPointer = FunnyPointer::from_data_ptr(&hello, &meta as *const _);
    }
}
