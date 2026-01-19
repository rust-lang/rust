// check-fail

use std::process::ExitCode;
use std::process::Termination;

trait IsStatic {}
impl<'a: 'static> IsStatic for &'a () {}

struct Thing;

impl Termination for Thing where for<'a> &'a (): IsStatic {
    fn report(self) -> ExitCode { panic!() }
}

fn main() -> Thing { Thing } //~ ERROR implementation of `IsStatic` is not general enough