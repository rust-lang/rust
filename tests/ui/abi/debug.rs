// normalize-stderr-test "(abi|pref|unadjusted_abi_align): Align\([1-8] bytes\)" -> "$1: $$SOME_ALIGN"
// normalize-stderr-test "(size): Size\([48] bytes\)" -> "$1: $$SOME_SIZE"
// normalize-stderr-test "(can_unwind): (true|false)" -> "$1: $$SOME_BOOL"
// normalize-stderr-test "(valid_range): 0\.\.=(4294967295|18446744073709551615)" -> "$1: $$FULL"
// This pattern is prepared for when we account for alignment in the niche.
// normalize-stderr-test "(valid_range): [1-9]\.\.=(429496729[0-9]|1844674407370955161[0-9])" -> "$1: $$NON_NULL"
// Some attributes are only computed for release builds:
// compile-flags: -O
#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[rustc_abi(debug)]
fn test(_x: u8) -> bool { true } //~ ERROR: fn_abi


#[rustc_abi(debug)]
fn test_generic<T>(_x: *const T) { } //~ ERROR: fn_abi

struct S(u16);
impl S {
    #[rustc_abi(debug)]
    fn assoc_test(&self) { } //~ ERROR: fn_abi
}
