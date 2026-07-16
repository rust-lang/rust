// An EII implementation cannot be `#[naked]`: EII forwards calls through a shim the
// compiler synthesizes around the body, which has nowhere to go in a naked function.
// See <https://github.com/rust-lang/rust/issues/158293>.

//@ needs-asm-support
#![feature(extern_item_impls)]
#![allow(dead_code)]

#[eii(requires_impl)]
unsafe extern "C" fn entry_eh(_: usize);

#[crate::requires_impl]
#[unsafe(naked)] //~ ERROR `#[requires_impl]` is not allowed to be `#[naked]`
unsafe extern "C" fn eh(_: usize) {
    core::arch::naked_asm!("nop");
}

#[eii(opt_impl)]
unsafe extern "C" fn entry_default(_: usize) {
    println!("called default")
}

#[crate::opt_impl]
#[unsafe(naked)] //~ ERROR `#[opt_impl]` is not allowed to be `#[naked]`
unsafe extern "C" fn eh_default(_: usize) {
    core::arch::naked_asm!("nop");
}

fn main() {}
