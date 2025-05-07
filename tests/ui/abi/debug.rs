//@ normalize-stderr: "(abi|pref|unadjusted_abi_align): Align\([1-8] bytes\)" -> "$1: $$SOME_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
//@ normalize-stderr: "(size): Size\([48] bytes\)" -> "$1: $$SOME_SIZE"
//@ normalize-stderr: "(can_unwind): (true|false)" -> "$1: $$SOME_BOOL"
//@ normalize-stderr: "(valid_range): 0\.\.=(4294967295|18446744073709551615)" -> "$1: $$FULL"
// This pattern is prepared for when we account for alignment in the niche.
//@ normalize-stderr: "(valid_range): [1-9]\.\.=(429496729[0-9]|1844674407370955161[0-9])" -> "$1: $$NON_NULL"
// Some attributes are only computed for release builds:
//@ compile-flags: -O
#![feature(rustc_attrs)]
#![crate_type = "lib"]

struct S(u16);

#[rustc_abi(debug)]
fn test(_x: u8) -> bool { true } //~ ERROR: fn_abi

#[rustc_abi(debug)]
type TestFnPtr = fn(bool) -> u8; //~ ERROR: fn_abi

#[rustc_abi(debug)]
fn test_generic<T>(_x: *const T) { } //~ ERROR: fn_abi

#[rustc_abi(debug)]
const C: () = (); //~ ERROR: can only be applied to

impl S {
    #[rustc_abi(debug)]
    const C: () = (); //~ ERROR: can only be applied to
}

impl S {
    #[rustc_abi(debug)]
    fn assoc_test(&self) { } //~ ERROR: fn_abi
}

#[rustc_abi(assert_eq)]
type TestAbiEq = (fn(bool), fn(bool));

#[rustc_abi(assert_eq)]
type TestAbiNe = (fn(u8), fn(u32)); //~ ERROR: ABIs are not compatible

#[rustc_abi(assert_eq)]
type TestAbiNeLarger = (fn([u8; 32]), fn([u32; 32])); //~ ERROR: ABIs are not compatible

#[rustc_abi(assert_eq)]
type TestAbiNeFloat = (fn(f32), fn(u32)); //~ ERROR: ABIs are not compatible

// Sign matters on some targets (such as s390x), so let's make sure we never accept this.
#[rustc_abi(assert_eq)]
type TestAbiNeSign = (fn(i32), fn(u32)); //~ ERROR: ABIs are not compatible

#[rustc_abi(assert_eq)]
type TestAbiEqNonsense = (fn((str, str)), fn((str, str))); //~ ERROR: cannot be known at compilation time

#[rustc_abi("assert_eq")] //~ ERROR unrecognized argument
type Bad = u32;
