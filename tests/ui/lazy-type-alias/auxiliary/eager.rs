// This crate does *not* have lazy type aliases enabled.

#![allow(type_alias_bounds)]

// The `Copy` bound is ignored both locally and externally for backward compatibility.
pub type Alias<T: Copy> = Option<T>;
