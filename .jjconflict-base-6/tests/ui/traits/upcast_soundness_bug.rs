//@ check-fail
//
// issue: <https://github.com/rust-lang/rust/pull/120222>
//! This would segfault at runtime.

pub trait SupSupA {
    fn method(&self) {}
}
pub trait SupSupB {}
impl<T> SupSupA for T {}
impl<T> SupSupB for T {}

pub trait Super<T>: SupSupA + SupSupB {}

pub trait Unimplemented {}

pub trait Trait<T1, T2>: Super<T1> + Super<T2> {
    fn missing_method(&self)
    where
        T1: Unimplemented,
    {
    }
}

impl<S, T> Super<T> for S {}

impl<S, T1, T2> Trait<T1, T2> for S {}

#[inline(never)]
pub fn user1() -> &'static dyn Trait<u8, u8> {
    &()
    /* VTABLE:
    .L__unnamed_2:
            .quad   core::ptr::drop_in_place<()>
            .asciz  "\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000"
            .quad   example::SupSupA::method
            .quad   .L__unnamed_4 // SupSupB vtable (pointer)
            .zero   8             // null pointer for missing_method
    */
}

#[inline(never)]
pub fn user2() -> &'static dyn Trait<u8, u16> {
    &()
    /* VTABLE:
    .L__unnamed_3:
            .quad   core::ptr::drop_in_place<()>
            .asciz  "\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000"
            .quad   example::SupSupA::method
            .quad   .L__unnamed_4 // SupSupB vtable (pointer)
            .quad   .L__unnamed_5 // Super<u16> vtable (pointer)
            .zero   8             // null pointer for missing_method
    */
}

fn main() {
    let p: *const dyn Trait<u8, u8> = &();
    let p = p as *const dyn Trait<u8, u16>; // <- this is bad!
    //~^ error: casting `*const dyn Trait<u8, u8>` as `*const dyn Trait<u8, u16>` is invalid
    let p = p as *const dyn Super<u16>; // <- this upcast accesses improper vtable entry
    // accessing from L__unnamed_2 the position for the 'Super<u16> vtable (pointer)',
    // thus reading 'null pointer for missing_method'

    let p = p as *const dyn SupSupB; // <- this upcast dereferences (null) pointer from that entry
    // to read the SupSupB vtable (pointer)

    // SEGFAULT

    println!("{:?}", p);
}
