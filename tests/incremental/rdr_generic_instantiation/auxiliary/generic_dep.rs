#![crate_name = "generic_dep"]
#![crate_type = "rlib"]

#[cfg(rpass1)]
fn private_helper<T>(x: T) -> T {
    x
}

#[cfg(any(rpass2, rpass3))]
fn private_helper<T>(x: T) -> T {
    let result = x;
    result
}

#[cfg(rpass3)]
fn _unused_generic<T>(_: T) {}

pub fn generic_fn<T: Copy>(x: T) -> T {
    private_helper(x)
}

pub struct GenericStruct<T> {
    pub value: T,
}

impl<T: Copy> GenericStruct<T> {
    pub fn get(&self) -> T {
        private_helper(self.value)
    }
}
