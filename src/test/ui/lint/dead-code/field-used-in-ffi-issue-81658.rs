//! The field `items` is being "used" by FFI (implicitly through pointers). However, since rustc
//! doesn't know how to detect that, we produce a message that says the field is unused. This can
//! cause some confusion and we want to make sure our diagnostics help as much as they can.
//!
//! Issue: https://github.com/rust-lang/rust/issues/81658

#![deny(dead_code)]

/// A struct for holding on to data while it is being used in our FFI code
pub struct FFIData<T> {
    /// These values cannot be dropped while the pointers to each item
    /// are still in use
    items: Option<Vec<T>>, //~ ERROR field is never read
}

impl<T> FFIData<T> {
    pub fn new() -> Self {
        Self {items: None}
    }

    /// Load items into this type and return pointers to each item that can
    /// be passed to FFI
    pub fn load(&mut self, items: Vec<T>) -> Vec<*const T> {
        let ptrs = items.iter().map(|item| item as *const _).collect();

        self.items = Some(items);

        ptrs
    }
}

extern {
    /// The FFI code that uses items
    fn process_item(item: *const i32);
}

fn main() {
    // Data cannot be dropped until the end of this scope or else the items
    // will be dropped before they are processed
    let mut data = FFIData::new();

    let ptrs = data.load(vec![1, 2, 3, 4, 5]);

    for ptr in ptrs {
        // Safety: This pointer is valid as long as the arena is in scope
        unsafe { process_item(ptr); }
    }

    // Items will be safely freed at the end of this scope
}
