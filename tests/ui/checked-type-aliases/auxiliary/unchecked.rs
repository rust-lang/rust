// This crate does *not* have checked type aliases enabled.
#![expect(type_alias_bounds)]

// The `Copy` bound is ignored both locally and externally for backward compatibility.
pub type Alias<T: Copy> = Option<T>;
