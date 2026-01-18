#![crate_name = "shared_dep"]
#![crate_type = "rlib"]

// Private generic helper - changes to this should not affect downstream
#[cfg(rpass1)]
fn private_helper<T>(x: T) -> T {
    x
}

#[cfg(any(rpass2, rpass3))]
fn private_helper<T>(x: T) -> T {
    let result = x;
    result
}

// Private non-generic function - changes should not affect downstream
#[cfg(any(rpass1, rpass2))]
fn private_fn() -> u32 {
    42
}

#[cfg(rpass3)]
fn private_fn() -> u32 {
    let x = 42;
    x
}

// Public generic function that uses the private helper
pub fn generic_fn<T: Copy>(x: T) -> T {
    let _ = private_fn();
    private_helper(x)
}

// Generic struct with drop glue to test share-generics drop handling
pub struct GenericBox<T> {
    value: T,
}

impl<T> GenericBox<T> {
    pub fn new(value: T) -> Self {
        GenericBox { value }
    }

    pub fn get(&self) -> &T {
        &self.value
    }
}

impl<T> Drop for GenericBox<T> {
    fn drop(&mut self) {
        // Drop glue is also shared with share-generics
    }
}
