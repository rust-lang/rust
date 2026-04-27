//! Check that we can't use `TypeId::trait_info_of` to unsoundly skip
//! the try_as_dyn checks.

#![feature(type_info, ptr_metadata, arbitrary_self_types_pointers)]

use std::any::TypeId;
use std::ptr::{self, DynMetadata};

type Payload = Box<i32>;

trait Trait {
    type Assoc;
    fn method(self: *const Self, value: Self::Assoc) -> &'static Payload;
}
struct Thing;
impl Trait for Thing {
    type Assoc = &'static Payload;
    fn method(self: *const Self, value: Self::Assoc) -> &'static Payload {
        value
    }
}

fn extend<'a>(payload: &'a Payload) -> &'static Payload {
    let metadata: DynMetadata<dyn Trait<Assoc = &'a Payload>> = const {
        TypeId::of::<Thing>()
            .trait_info_of::<dyn Trait<Assoc = &'a Payload>>()
            //~^ ERROR `dyn Trait<Assoc = &'a Box<i32>>: TryAsDynCompatible<'_>` is not satisfied
            .unwrap()
            .get_vtable()
    };
    let ptr: *const dyn Trait<Assoc = &'a Payload> =
        ptr::from_raw_parts(std::ptr::null::<()>(), metadata);
    ptr.method(payload)
}

fn main() {
    let payload: Box<Payload> = Box::new(Box::new(1i32));
    let wrong: &'static Payload = extend(&*payload);
    drop(payload);
    println!("{wrong}");
}
