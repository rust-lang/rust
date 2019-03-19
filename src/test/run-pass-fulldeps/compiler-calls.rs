// Test that the Callbacks interface to the compiler works.

// ignore-cross-compile
// ignore-stage1

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;

use rustc_interface::interface;

struct TestCalls<'a> {
    count: &'a mut u32
}

impl rustc_driver::Callbacks for TestCalls<'_> {
    fn config(&mut self, _config: &mut interface::Config) {
        *self.count *= 2;
    }
}

fn main() {
    let mut count = 1;
    let args = vec!["compiler-calls".to_string(), "foo.rs".to_string()];
    rustc_driver::report_ices_to_stderr_if_any(|| {
        rustc_driver::run_compiler(&args, &mut TestCalls { count: &mut count }, None, None).ok();
    }).ok();
    assert_eq!(count, 2);
}
