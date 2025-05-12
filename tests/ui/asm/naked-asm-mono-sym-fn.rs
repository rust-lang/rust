// Regression test for <https://github.com/rust-lang/rust/issues/140373>.
// Test that we're properly monomorphizing sym args in naked asm blocks
// that point to associated items.

//@ edition: 2021
//@ needs-asm-support
//@ only-x86_64
//@ build-pass

trait Tr {
    extern "C" fn t();
}

enum E<const C: usize> {}

impl<const C: usize> Tr for E<C> {
    extern "C" fn t() {
        println!("Const generic: {}", C);
    }
}

#[unsafe(naked)]
extern "C" fn foo<U: Tr>() {
    core::arch::naked_asm!(
        "push rax",
        "call {fn}",
        "pop rax",
        "ret",
        fn = sym <U as Tr>::t,
    );
}

fn main() {
    foo::<E<42>>();
}
