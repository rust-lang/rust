//@ build-pass
//@ needs-asm-support

#![feature(asm_const_ptr)]

// Force inline to exercise the codegen when the same asm const ptr is code-generated multiple
// times.
#[inline(always)]
fn foo<const N:usize>() {
    unsafe{core::arch::asm!("/* {} */", const &N)};
}

fn main(){
    foo::<0>();
    foo::<0>();
    foo::<1>();
}
