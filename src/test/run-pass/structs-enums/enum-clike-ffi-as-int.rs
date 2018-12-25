// run-pass
#![allow(dead_code)]

/*!
 * C-like enums have to be represented as LLVM ints, not wrapped in a
 * struct, because it's important for the FFI that they interoperate
 * with C integers/enums, and the ABI can treat structs differently.
 * For example, on i686-linux-gnu, a struct return value is passed by
 * storing to a hidden out parameter, whereas an integer would be
 * returned in a register.
 *
 * This test just checks that the ABIs for the enum and the plain
 * integer are compatible, rather than actually calling C code.
 * The unused parameter to `foo` is to increase the likelihood of
 * crashing if something goes wrong here.
 */

#[repr(u32)]
enum Foo {
    A = 0,
    B = 23
}

#[inline(never)]
extern "C" fn foo(_x: usize) -> Foo { Foo::B }

pub fn main() {
    unsafe {
        let f: extern "C" fn(usize) -> u32 =
            ::std::mem::transmute(foo as extern "C" fn(usize) -> Foo);
        assert_eq!(f(0xDEADBEEF), Foo::B as u32);
    }
}
