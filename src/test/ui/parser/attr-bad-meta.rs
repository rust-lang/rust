// compile-flags: -Z parse-only

#![feature(unrestricted_attribute_tokens)]

#[path*] //~ ERROR expected one of `(`, `::`, `=`, `[`, `]`, or `{`, found `*`
mod m {}
