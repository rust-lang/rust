// Compiler:

// `Set` reaches itself by value through `*mut Root`, so a backend that emits `Root`'s
// fields with the still-incomplete `Set` type fails to compile this.

#![crate_type = "lib"]

#[repr(C)]
pub struct Set {
    pub root: *mut Root,
    pub first: usize,
    pub second: usize,
}

#[repr(C)]
pub struct Root {
    pub default_set: Set,
}

pub fn identity(set: Set) -> Set {
    set
}
