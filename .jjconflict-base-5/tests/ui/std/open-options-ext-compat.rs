//@ only-windows
//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/153486
// Ensures that `OpenOptionsExt` remains implementable by downstream crates
// without requiring changes when new methods are added to the standard library.

use std::os::windows::fs::OpenOptionsExt;

struct MockOptions;

impl OpenOptionsExt for MockOptions {
    fn access_mode(&mut self, _: u32) -> &mut Self { self }
    fn share_mode(&mut self, _: u32) -> &mut Self { self }
    fn custom_flags(&mut self, _: u32) -> &mut Self { self }
    fn attributes(&mut self, _: u32) -> &mut Self { self }
    fn security_qos_flags(&mut self, _: u32) -> &mut Self { self }
}

fn main() {}
