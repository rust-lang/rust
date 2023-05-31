// build-pass
// regression test for #112051

#![feature(offset_of)]

struct S<T: ?Sized> {
    a: u64,
    b: T,
}
trait Tr {}

fn main() {
    let _a = core::mem::offset_of!(S<dyn Tr>, a);
    let _b = core::mem::offset_of!((u64, dyn Tr), 0);
}
