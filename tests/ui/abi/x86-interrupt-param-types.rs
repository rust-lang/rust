//! Tests x86 interrupt ABI parameter validation
//! Specifically, we don't test the arity of the parameters (that is handled
//!   by `interrupt-invalid-signature.rs`), but the shape of the parameters.
//!  In short:
//!    Frames:
//!    1. Reference or value
//!    2. Complete over all possible bit patterns
//!    3. Neither ZST or unsized
//!    Codes:
//!    1. Presence optional
//!    2. Machine sized scalar, complete over all possible bit patterns.
//@ add-minicore
//@ revisions: x64 i686
//
//@ [x64] needs-llvm-components: x86
//@ [x64] compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
//@ [i686] needs-llvm-components: x86
//@ [i686] compile-flags: --target=i686-unknown-linux-gnu --crate-type=rlib
//@ ignore-backends: gcc
#![no_core]
#![feature(
    no_core,
    abi_x86_interrupt,
    unsized_fn_params
)]

extern crate minicore;
use minicore::*;

#[repr(C)]
struct Frame {
    ip: u64,
    cs: u64,
    flags: u64,
    sp: u64,
    ss: u64
}

#[repr(C)]
struct NonScalar32 {
    a: u8,
    b: u16
}

#[repr(C)]
struct NonScalar64 {
    a: u8,
    b: u32
}

#[repr(transparent)]
struct Newtype(usize);

#[repr(u64)]
enum Enumtype {
    A
}

enum Opt<T> {
    Some(T),
    None
}

#[repr(C)]
struct NewBool{ a: bool }

/* Frame parameter tests */
extern "x86-interrupt" fn test_frame_unsized_fails(_: str) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_zst_fails(_: ()) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_bool_fails(_: bool) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_char_fails(_: char) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_indirect_bool_fails(_: NewBool) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_pointer_fails(_: *const u8) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_mut_pointer_fails(_: *mut u8) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_reference_fails(_: &Frame) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_mut_reference_fails(_: &mut Frame) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_optional_reference_fails(_: Opt<&Frame>) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_frame_optional_mut_reference_fails(_: Opt<&mut Frame>) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function

extern "x86-interrupt" fn test_frame_actual_frame_works(_: Frame) {}

/* Error code parameter tests */
extern "x86-interrupt" fn test_code_unsized_fails(_: Frame, _: str) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_zst_fails(_: Frame, _: ()) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_u8_fails(_: Frame, _: u8) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_u16_fails(_: Frame, _: u16) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
#[cfg(i686)]
extern "x86-interrupt" fn test_code_32bit_u64_fails(_: Frame, _: u64) {}
//[i686]~^ ERROR invalid signature for `extern "x86-interrupt"` function
#[cfg(x64)]
extern "x86-interrupt" fn test_code_64bit_u32_fails(_: Frame, _: u32) {}
//[x64]~^ ERROR invalid signature for `extern "x86-interrupt"` function
#[cfg(i686)]
extern "x86-interrupt" fn test_code_non_scalar_32_fails(_: Frame, _: NonScalar32) {}
//[i686]~^ ERROR invalid signature for `extern "x86-interrupt"` function
#[cfg(x64)]
extern "x86-interrupt" fn test_code_non_scalar_64_fails(_: Frame, _: NonScalar64) {}
//[x64]~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_pointer_fails(_: Frame, _: *const u8) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_mut_pointer_fails(_: Frame, _: *mut u8) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_u128_fails(_: Frame, _: u128) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_bool_fails(_: Frame, _: bool) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_array_fails(_: Frame, _: [u8; 3]) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_enum_fails(_: Frame, _: Enumtype) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function
#[cfg(x64)]
extern "x86-interrupt" fn test_code_64bit_float_fails(_: Frame, _: f64) {}
//[x64]~^ ERROR invalid signature for `extern "x86-interrupt"` function
#[cfg(i686)]
extern "x86-interrupt" fn test_code_32bit_float_fails(_: Frame, _: f32) {}
//[i686]~^ ERROR invalid signature for `extern "x86-interrupt"` function
extern "x86-interrupt" fn test_code_char_fails(_: Frame, _: char) {}
//~^ ERROR invalid signature for `extern "x86-interrupt"` function

#[cfg(i686)]
extern "x86-interrupt" fn test_code_32bit_u32_works(_: Frame, _: u32) {}
#[cfg(x64)]
extern "x86-interrupt" fn test_code_64bit_u64_works(_: Frame, _: u64) {}
extern "x86-interrupt" fn test_code_usize_works(_: Frame, _: usize) {}
extern "x86-interrupt" fn test_code_newtype_works(_: Frame, _: Newtype) {}
extern "x86-interrupt" fn test_code_signedness_works(_: Frame, _: isize) {}

/* Impl shape tests */
#[repr(C)]
struct FancyFrame {
    ip: u64,
    cs: u64,
    flags: u64,
    sp: u64,
    ss: u64
}

impl FancyFrame {

    extern "x86-interrupt" fn test_impl_frame_zst_fails(_: ()) {}
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function
    extern "x86-interrupt" fn test_impl_frame_unsized_fails(_: str) {}
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function
    extern "x86-interrupt" fn test_impl_code_zst_fails(_: Frame, _: ()) {}
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function
    extern "x86-interrupt" fn test_impl_code_unsized_fails(_: Frame, _: str) {}
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function

    extern "x86-interrupt" fn test_impl_self_works(_: Self) {}
    extern "x86-interrupt" fn test_impl_frame_works(_: Frame) {}
    extern "x86-interrupt" fn test_impl_code_works(_: Frame, _: usize) {}
}

/* Trait shape tests */
trait Handler {
    extern "x86-interrupt" fn test_trait_impl_passes(_: Self);
}

/* Foreign decl test */
unsafe extern "x86-interrupt" {
    fn test_extern_frame_zst_fails(_: ());
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function
    fn test_extern_frame_unsized_fails(_: str);
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function
    fn test_extern_code_zst_fails(_: Frame, _: ());
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function
    fn test_extern_code_unsized_fails(_: Frame, _: str);
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function
    fn test_extern_code_enum_fails(_: Frame, _: Enumtype);
    //~^ ERROR invalid signature for `extern "x86-interrupt"` function

    fn test_extern_frame_works(_: Frame);
    fn test_extern_code_works(_: Frame, _: usize);
    fn test_extern_code_newtype_works(_: Frame, _: Newtype);
}

/* Complex frame tests */
struct Addr(u64);

#[repr(C)]
struct ComplexFrame { ip: Addr, cs: u64, flags: u64, sp: Addr, ss: u64 }

struct Idt { function_pointer_member_should_work: extern "x86-interrupt" fn(ComplexFrame) }

/* Generic tests */
trait WithType {
    type Associated;

    extern "x86-interrupt" fn test_trait_default_works(_: Self) {}
}

struct GenericFrame<T: Copy> {
    ip: T,
    cs: T,
    flags: T,
    sp: T,
    ss: T
}

impl<T: Copy> GenericFrame<T> {
    extern "x86-interrupt" fn test_generic_impl_works(_: Self) {}
}

#[repr(transparent)]
struct GenericNewType<T: Copy>(T);

extern "x86-interrupt" fn test_const_param_generic_works<const N: usize>(_: [u64; N]) {}
extern "x86-interrupt" fn test_body_generics_works<const N: usize>(_: Frame) {}
extern "x86-interrupt" fn test_param_generics_frame_works<F>(_: F) {}
extern "x86-interrupt" fn test_param_generics_code_works<F, E>(_: F, _: E) {}
extern "x86-interrupt" fn test_param_generics_wrapper_works<F>(_: Opt<F>) {}
extern "x86-interrupt" fn test_param_generics_assoc_type_works<F: WithType>(_: F::Associated) {}
extern "x86-interrupt" fn test_param_impl_generics_works(_: impl Copy) {}
extern "x86-interrupt" fn test_param_impl_newtype_works(_: Frame, _: GenericNewType<usize>) {}
