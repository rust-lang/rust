//@ revisions: next old
//@[next] compile-flags: -Znext-solver
#![feature(try_as_dyn)]

use std::any::try_as_dyn;

type Payload = *const i32;

trait Convert<T> {
    fn convert(&self) -> &T;
}

impl<T> Convert<T> for T {
    fn convert(&self) -> &T {
        self
    }
}

const _: () = {
    let payload: Payload = std::ptr::null();
    let convert: &dyn Convert<&'static Payload> = try_as_dyn(&payload).unwrap();
    //~^ ERROR: `Option::unwrap()` on a `None` value
};

fn main() {}
