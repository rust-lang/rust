use std::alloc::Layout;
use std::any::TypeId;
use std::mem::transmute;
use std::ptr::drop_in_place;

pub struct VTable {
    layout: Layout,
    type_id: TypeId,
    drop_in_place: unsafe fn(*mut ()),
}

impl VTable {
    pub fn new<T>() -> &'static Self {
        const {
            &VTable {
                layout: Layout::new::<T>(),
                type_id: TypeId::of::<T>(),
                //~^ ERROR the parameter type `T` may not live long enough
                //~| ERROR the parameter type `T` may not live long enough
                drop_in_place: unsafe {
                    transmute::<unsafe fn(*mut T), unsafe fn(*mut ())>(drop_in_place::<T>)
                },
            }
        }
    }
}

fn main() {}
