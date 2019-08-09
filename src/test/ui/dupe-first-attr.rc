// Regression test for a problem with the first mod attribute
// being applied to every mod

// pretty-expanded FIXME #23616

#[cfg(target_os = "linux")]
mod hello;

#[cfg(target_os = "macos")]
mod hello;

#[cfg(target_os = "windows")]
mod hello;

#[cfg(target_os = "freebsd")]
mod hello;

#[cfg(target_os = "dragonfly")]
mod hello;

#[cfg(target_os = "android")]
mod hello;

pub fn main() { }
