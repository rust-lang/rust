// Verifies codeview_annotation compile-time behavior:
// - Happy paths: inline literals, macro usage, named const elements,
//   mixed const/literal, const slices, const array refs, and const fn usage.
// - Error cases: non-const arguments, function parameters, empty arrays,
//   and wrong types.
#![feature(codeview_annotation)]
#![feature(core_intrinsics)]

use std::intrinsics::codeview_annotation;

const STR_A: &str = "string1";
const STR_B: &str = "string2";
const STR_C: &str = "string3";

const STRS_SLICE: &[&str] = &["string1", "string2", "string3"];
const STRS_ARRAY: [&str; 3] = ["string1", "string2", "string3"];

trait Var {
    const NAME: &str;
    const VAL: &str;
}

impl Var for i32 {
    const NAME: &str = "i32";
    const VAL: &str = "5";
}

// === Intrinsic tests ===

fn single() {
    codeview_annotation(&["string1"]);
}

fn multiple() {
    codeview_annotation(&["string1", "string2", "string3"]);
}

fn named_const_elements() {
    codeview_annotation(&[STR_A, STR_B, STR_C]);
}

fn mixed_named_const_and_literal_elements() {
    codeview_annotation(&[STR_A, "string2", "string3"]);
}

fn named_const_slice() {
    codeview_annotation(STRS_SLICE);
}

fn named_const_array_ref() {
    codeview_annotation(&STRS_ARRAY);
}

// Use in const function
const fn const_func(x: u32) -> u32 {
    codeview_annotation(&["string1", "string2", "string3"]);
    x + 1
}

fn generic_const_elements<T: Var>() {
    codeview_annotation(&["string", T::NAME, T::VAL]);
}

// --- Error cases ---

fn non_const_arg(strs: &[&str]) {
    codeview_annotation(strs); //~ ERROR `codeview_annotation` expects a const array
    let s = "string1";
    codeview_annotation(&[s]); //~ ERROR `codeview_annotation` expects a const array
}

fn empty_array() {
    codeview_annotation(&[]); //~ ERROR `codeview_annotation` expects a non-empty array
}

fn wrong_type() {
    codeview_annotation(42); //~ ERROR mismatched types
}

// Slices that are associated consts on generic types are not
// yet supported because they complicate the implementation
trait HasStrs {
    const STRS: &[&str];
}

impl HasStrs for i32 {
    const STRS: &[&str] = &["string1", "string2", "string3"];
}

fn generic_associated_const_slice<T: HasStrs>() {
    codeview_annotation(T::STRS); //~ ERROR `codeview_annotation` argument cannot be a generic const
}



// === Macro tests ===

fn macro_single() {
    std::hint::codeview_annotation!("string1");
}

fn macro_multiple() {
    std::hint::codeview_annotation!("string1", "string2", "string3");
}

fn macro_named_const_elements() {
    std::hint::codeview_annotation!(STR_A, STR_B, STR_C);
}

fn macro_mixed_named_const_and_literal_elements() {
    std::hint::codeview_annotation!(STR_A, "string2", "string3");
}

const fn macro_const_func(x: u32) -> u32 {
    std::hint::codeview_annotation!("string1", "string2", "string3");
    x + 1
}

fn macro_generic_const_elements<T: Var>() {
    std::hint::codeview_annotation!("string", T::NAME, T::VAL);
}

// --- Error cases ---

fn macro_non_const_arg() {
    let s = "string1";
    std::hint::codeview_annotation!(s); //~ ERROR `codeview_annotation` expects a const array
}

fn macro_wrong_type() {
    std::hint::codeview_annotation!(42); //~ ERROR mismatched types
}

fn macro_generic_associated_const_slice<T: HasStrs>() {
    std::hint::codeview_annotation!(T::STRS); //~ ERROR mismatched types
}


fn main() {
    single();
    multiple();
    named_const_elements();
    mixed_named_const_and_literal_elements();
    named_const_slice();
    named_const_array_ref();
    let _ = const_func(5);
    const _: u32 = const_func(5);
    generic_const_elements::<i32>();

    non_const_arg(&["a"]);
    empty_array();
    wrong_type();
    generic_associated_const_slice::<i32>();


    macro_single();
    macro_multiple();
    macro_named_const_elements();
    macro_mixed_named_const_and_literal_elements();
    let _ = macro_const_func(5);
    const _: u32 = macro_const_func(5);
    macro_generic_const_elements::<i32>();

    macro_non_const_arg();
    macro_wrong_type();
    macro_generic_associated_const_slice::<i32>();
}
