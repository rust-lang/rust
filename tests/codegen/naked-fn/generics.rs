//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "lib"]

use std::arch::naked_asm;

#[no_mangle]
fn test(x: u64) {
    // just making sure these symbols get used
    using_const_generics::<1>(x);
    using_const_generics::<2>(x);

    generic_function::<i64>(x as i64);

    let foo = Foo(x);

    foo.method();
    foo.trait_method();
}

// CHECK: .balign 4
// CHECK: add rax, 2
// CHECK: add rax, 42

// CHECK: .balign 4
// CHECK: add rax, 1
// CHECK: add rax, 42

#[unsafe(naked)]
pub extern "C" fn using_const_generics<const N: u64>(x: u64) -> u64 {
    const M: u64 = 42;

    naked_asm!(
        "xor rax, rax",
        "add rax, rdi",
        "add rax, {}",
        "add rax, {}",
        "ret",
        const N,
        const M,
    )
}

trait Invert {
    fn invert(self) -> Self;
}

impl Invert for i64 {
    fn invert(self) -> Self {
        -1 * self
    }
}

// CHECK: .balign 4
// CHECK-LABEL: generic_function:
// CHECK: call
// CHECK: ret

#[unsafe(naked)]
#[no_mangle]
pub extern "C" fn generic_function<T: Invert>(x: i64) -> i64 {
    naked_asm!(
        "call {}",
        "ret",
        sym <T as Invert>::invert,
    )
}

#[derive(Copy, Clone)]
#[repr(transparent)]
struct Foo(u64);

// CHECK: .balign 4
// CHECK-LABEL: method:
// CHECK: mov rax, rdi

impl Foo {
    #[unsafe(naked)]
    #[no_mangle]
    extern "C" fn method(self) -> u64 {
        naked_asm!("mov rax, rdi", "ret")
    }
}

// CHECK: .balign 4
// CHECK-LABEL: trait_method:
// CHECK: mov rax, rdi

trait Bar {
    extern "C" fn trait_method(self) -> u64;
}

impl Bar for Foo {
    #[unsafe(naked)]
    #[no_mangle]
    extern "C" fn trait_method(self) -> u64 {
        naked_asm!("mov rax, rdi", "ret")
    }
}

// CHECK: .balign 4
// CHECK-LABEL: naked_with_args_and_return:
// CHECK: lea rax, [rdi + rsi]

// this previously ICE'd, see https://github.com/rust-lang/rust/issues/124375
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn naked_with_args_and_return(a: isize, b: isize) -> isize {
    naked_asm!("lea rax, [rdi + rsi]", "ret");
}
