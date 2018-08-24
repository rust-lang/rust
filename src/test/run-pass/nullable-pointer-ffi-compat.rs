// #11303, #11040:
// This would previously crash on i686 Linux due to abi differences
// between returning an Option<T> and T, where T is a non nullable
// pointer.
// If we have an enum with two variants such that one is zero sized
// and the other contains a nonnullable pointer, we don't use a
// separate discriminant. Instead we use that pointer field to differentiate
// between the 2 cases.
// Also, if the variant with the nonnullable pointer has no other fields
// then we simply express the enum as just a pointer and not wrap it
// in a struct.


use std::mem;

#[inline(never)]
extern "C" fn foo(x: &isize) -> Option<&isize> { Some(x) }

static FOO: isize = 0xDEADBEE;

pub fn main() {
    unsafe {
        let f: extern "C" fn(&isize) -> &isize =
            mem::transmute(foo as extern "C" fn(&isize) -> Option<&isize>);
        assert_eq!(*f(&FOO), FOO);
    }
}
