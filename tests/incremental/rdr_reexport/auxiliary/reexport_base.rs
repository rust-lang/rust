#![crate_name = "reexport_base"]
#![crate_type = "rlib"]

#[cfg(rpass1)]
fn private_impl() -> u32 {
    42
}

#[cfg(any(rpass2, rpass3))]
fn private_impl() -> u32 {
    21 + 21
}

pub struct Thing {
    value: u32,
}

impl Thing {
    pub fn new() -> Self {
        Thing { value: private_impl() }
    }

    pub fn get(&self) -> u32 {
        self.value
    }
}

pub fn create_thing() -> Thing {
    Thing::new()
}
