//@no-rustfix
#![deny(clippy::should_panic_without_expect)]

#[test]
#[should_panic]
fn no_message() {}

#[test]
#[should_panic]
#[cfg(not(test))]
fn no_message_cfg_false() {}

#[test]
#[should_panic = "message"]
fn metastr() {}

#[test]
#[should_panic(expected = "message")]
fn metalist() {}

fn main() {}
