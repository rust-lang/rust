#![crate_name = "impl_trait_dep"]
#![crate_type = "rlib"]

pub trait MyTrait {
    fn value(&self) -> u32;
}

#[cfg(rpass1)]
struct PrivateImpl {
    x: u32,
}

#[cfg(any(rpass2, rpass3))]
struct PrivateImpl {
    x: u32,
    _extra: u32,
}

impl MyTrait for PrivateImpl {
    fn value(&self) -> u32 {
        self.x
    }
}

#[cfg(rpass1)]
pub fn make_thing() -> impl MyTrait {
    PrivateImpl { x: 42 }
}

#[cfg(any(rpass2, rpass3))]
pub fn make_thing() -> impl MyTrait {
    PrivateImpl { x: 42, _extra: 0 }
}

#[cfg(rpass3)]
struct _AnotherPrivate;
