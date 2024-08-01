// This module contains code that is shared between all platforms, mostly utility or fallback code.
// This explicitly does not include code that is shared between only a few platforms,
// such as when reusing an implementation from `unix` or `unsupported`.
// In those cases the desired code should be included directly using the #[path] attribute,
// not moved to this module.
//
// Currently `sys_common` contains a lot of code that should live in this module,
// ideally `sys_common` would only contain platform-independent abstractions on top of `sys`.
// Progress on this is tracked in #84187.

#![allow(dead_code)]

pub mod small_c_string;

#[cfg(test)]
mod tests;
