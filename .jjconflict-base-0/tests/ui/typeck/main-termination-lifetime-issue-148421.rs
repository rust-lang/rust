// This test checks that the compiler correctly handles lifetime requirements
// on the `Termination` trait for the `main` function return type.
// See https://github.com/rust-lang/rust/issues/148421

use std::process::ExitCode;
use std::process::Termination;

trait IsStatic {}
impl<'a: 'static> IsStatic for &'a () {}

struct Thing;

impl Termination for Thing where for<'a> &'a (): IsStatic {
    fn report(self) -> ExitCode { panic!() }
}

fn main() -> Thing { Thing } //~ ERROR implementation of `IsStatic` is not general enough
