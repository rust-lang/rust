//@aux-build:../../ui/auxiliary/proc_macro_unsafe.rs
//@revisions: default disabled
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/undocumented_unsafe_blocks/default
//@[disabled] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/undocumented_unsafe_blocks/disabled

#![warn(clippy::undocumented_unsafe_blocks, clippy::unnecessary_safety_comment)]
#![allow(
    deref_nullptr,
    non_local_definitions,
    clippy::let_unit_value,
    clippy::missing_safety_doc
)]

extern crate proc_macro_unsafe;

// Valid comments

fn nested_local() {
    let _ = {
        let _ = {
            // SAFETY:
            let _ = unsafe {};
        };
    };
}

fn deep_nest() {
    let _ = {
        let _ = {
            // SAFETY:
            let _ = unsafe {};

            // Safety:
            unsafe {};

            let _ = {
                let _ = {
                    let _ = {
                        let _ = {
                            let _ = {
                                // Safety:
                                let _ = unsafe {};

                                // SAFETY:
                                unsafe {};
                            };
                        };
                    };

                    // Safety:
                    unsafe {};
                };
            };
        };

        // Safety:
        unsafe {};
    };

    // SAFETY:
    unsafe {};
}

fn local_tuple_expression() {
    // Safety:
    let _ = (42, unsafe {});
}

fn line_comment() {
    // Safety:
    unsafe {}
}

fn line_comment_newlines() {
    // SAFETY:

    unsafe {}
}

fn line_comment_empty() {
    // Safety:
    //
    //
    //
    unsafe {}
}

fn line_comment_with_extras() {
    // This is a description
    // Safety:
    unsafe {}
}

fn block_comment() {
    /* Safety: */
    unsafe {}
}

fn block_comment_newlines() {
    /* SAFETY: */

    unsafe {}
}

fn block_comment_with_extras() {
    /* This is a description
     * SAFETY:
     */
    unsafe {}
}

fn block_comment_terminator_same_line() {
    /* This is a description
     * Safety: */
    unsafe {}
}

fn buried_safety() {
    // Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
    // incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation
    // ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in
    // reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
    // occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est
    // laborum. Safety:
    // Tellus elementum sagittis vitae et leo duis ut diam quam. Sit amet nulla facilisi
    // morbi tempus iaculis urna. Amet luctus venenatis lectus magna. At quis risus sed vulputate odio
    // ut. Luctus venenatis lectus magna fringilla urna. Tortor id aliquet lectus proin nibh nisl
    // condimentum id venenatis. Vulputate dignissim suspendisse in est ante in nibh mauris cursus.
    unsafe {}
}

fn safety_with_prepended_text() {
    // This is a test. safety:
    unsafe {}
}

fn local_line_comment() {
    // Safety:
    let _ = unsafe {};
}

fn local_block_comment() {
    /* SAFETY: */
    let _ = unsafe {};
}

fn comment_array() {
    // Safety:
    let _ = [unsafe { 14 }, unsafe { 15 }, 42, unsafe { 16 }];
}

fn comment_tuple() {
    // sAFETY:
    let _ = (42, unsafe {}, "test", unsafe {});
}

fn comment_unary() {
    // SAFETY:
    let _ = *unsafe { &42 };
}

#[allow(clippy::match_single_binding)]
fn comment_match() {
    // SAFETY:
    let _ = match unsafe {} {
        _ => {},
    };
}

fn comment_addr_of() {
    // Safety:
    let _ = &unsafe {};
}

fn comment_repeat() {
    // Safety:
    let _ = [unsafe {}; 5];
}

fn comment_macro_call() {
    macro_rules! t {
        ($b:expr) => {
            $b
        };
    }

    t!(
        // SAFETY:
        unsafe {}
    );
}

fn comment_macro_def() {
    macro_rules! t {
        () => {
            // Safety:
            unsafe {}
        };
    }

    t!();
}

fn non_ascii_comment() {
    // ॐ᧻໒ SaFeTy: ௵∰
    unsafe {};
}

fn local_commented_block() {
    let _ =
        // safety:
        unsafe {};
}

fn local_nest() {
    // safety:
    let _ = [(42, unsafe {}, unsafe {}), (52, unsafe {}, unsafe {})];
}

fn in_fn_call(x: *const u32) {
    fn f(x: u32) {}

    // Safety: reason
    f(unsafe { *x });
}

fn multi_in_fn_call(x: *const u32) {
    fn f(x: u32, y: u32) {}

    // Safety: reason
    f(unsafe { *x }, unsafe { *x });
}

fn in_multiline_fn_call(x: *const u32) {
    fn f(x: u32, y: u32) {}

    f(
        // Safety: reason
        unsafe { *x },
        0,
    );
}

fn in_macro_call(x: *const u32) {
    // Safety: reason
    println!("{}", unsafe { *x });
}

fn in_multiline_macro_call(x: *const u32) {
    println!(
        "{}",
        // Safety: reason
        unsafe { *x },
    );
}

fn from_proc_macro() {
    proc_macro_unsafe::unsafe_block!(token);
}

fn in_closure(x: *const u32) {
    // Safety: reason
    let _ = || unsafe { *x };
}

// Invalid comments

#[rustfmt::skip]
fn inline_block_comment() {
    /* Safety: */ unsafe {}
    //~^ undocumented_unsafe_blocks
}

fn no_comment() {
    unsafe {}
    //~^ undocumented_unsafe_blocks
}

fn no_comment_array() {
    let _ = [unsafe { 14 }, unsafe { 15 }, 42, unsafe { 16 }];
    //~^ undocumented_unsafe_blocks
    //~| undocumented_unsafe_blocks
    //~| undocumented_unsafe_blocks
}

fn no_comment_tuple() {
    let _ = (42, unsafe {}, "test", unsafe {});
    //~^ undocumented_unsafe_blocks
    //~| undocumented_unsafe_blocks
}

fn no_comment_unary() {
    let _ = *unsafe { &42 };
    //~^ undocumented_unsafe_blocks
}

#[allow(clippy::match_single_binding)]
fn no_comment_match() {
    let _ = match unsafe {} {
        //~^ undocumented_unsafe_blocks
        _ => {},
    };
}

fn no_comment_addr_of() {
    let _ = &unsafe {};
    //~^ undocumented_unsafe_blocks
}

fn no_comment_repeat() {
    let _ = [unsafe {}; 5];
    //~^ undocumented_unsafe_blocks
}

fn local_no_comment() {
    let _ = unsafe {};
    //~^ undocumented_unsafe_blocks
}

fn no_comment_macro_call() {
    macro_rules! t {
        ($b:expr) => {
            $b
        };
    }

    t!(unsafe {});
    //~^ undocumented_unsafe_blocks
}

fn no_comment_macro_def() {
    macro_rules! t {
        () => {
            unsafe {}
            //~^ undocumented_unsafe_blocks
        };
    }

    t!();
}

fn trailing_comment() {
    unsafe {} // SAFETY:
    //
    //~^^ undocumented_unsafe_blocks
}

fn internal_comment() {
    unsafe {
        //~^ undocumented_unsafe_blocks
        // SAFETY:
    }
}

fn interference() {
    // SAFETY

    let _ = 42;

    unsafe {};
    //~^ undocumented_unsafe_blocks
}

pub fn print_binary_tree() {
    println!("{}", unsafe { String::from_utf8_unchecked(vec![]) });
    //~^ undocumented_unsafe_blocks
}

mod unsafe_impl_smoke_test {
    unsafe trait A {}

    // error: no safety comment
    unsafe impl A for () {}
    //~^ undocumented_unsafe_blocks

    // Safety: ok
    unsafe impl A for (i32) {}

    mod sub_mod {
        // error:
        unsafe impl B for (u32) {}
        //~^ undocumented_unsafe_blocks
        unsafe trait B {}
    }

    #[rustfmt::skip]
    mod sub_mod2 {
        //
        // SAFETY: ok
        //

        unsafe impl B for (u32) {}
        unsafe trait B {}
    }
}

mod unsafe_impl_from_macro {
    unsafe trait T {}

    // error
    macro_rules! no_safety_comment {
        ($t:ty) => {
            unsafe impl T for $t {}
            //~^ undocumented_unsafe_blocks
        };
    }

    // ok
    no_safety_comment!(());

    // ok
    macro_rules! with_safety_comment {
        ($t:ty) => {
            // SAFETY:
            unsafe impl T for $t {}
        };
    }

    // ok
    with_safety_comment!((i32));
}

mod unsafe_impl_macro_and_not_macro {
    unsafe trait T {}

    // error
    macro_rules! no_safety_comment {
        ($t:ty) => {
            unsafe impl T for $t {}
            //~^ undocumented_unsafe_blocks
            //~| undocumented_unsafe_blocks
        };
    }

    // ok
    no_safety_comment!(());

    // error
    unsafe impl T for (i32) {}
    //~^ undocumented_unsafe_blocks

    // ok
    no_safety_comment!(u32);

    // error
    unsafe impl T for (bool) {}
    //~^ undocumented_unsafe_blocks
}

#[rustfmt::skip]
mod unsafe_impl_valid_comment {
    unsafe trait SaFety {}
    // SaFety:
    unsafe impl SaFety for () {}

    unsafe trait MultiLineComment {}
    // The following impl is safe
    // ...
    // Safety: reason
    unsafe impl MultiLineComment for () {}

    unsafe trait NoAscii {}
    // 安全 SAFETY: 以下のコードは安全です
    unsafe impl NoAscii for () {}

    unsafe trait InlineAndPrecedingComment {}
    // SAFETY:
    /* comment */ unsafe impl InlineAndPrecedingComment for () {}

    unsafe trait BuriedSafety {}
    // Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
    // incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation
    // ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in
    // reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
    // occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est
    // laborum. Safety:
    // Tellus elementum sagittis vitae et leo duis ut diam quam. Sit amet nulla facilisi
    // morbi tempus iaculis urna. Amet luctus venenatis lectus magna. At quis risus sed vulputate odio
    // ut. Luctus venenatis lectus magna fringilla urna. Tortor id aliquet lectus proin nibh nisl
    // condimentum id venenatis. Vulputate dignissim suspendisse in est ante in nibh mauris cursus.
    unsafe impl BuriedSafety for () {}

    unsafe trait MultiLineBlockComment {}
    /* This is a description
     * Safety: */
    unsafe impl MultiLineBlockComment for () {}
}

#[rustfmt::skip]
mod unsafe_impl_invalid_comment {
    unsafe trait NoComment {}

    unsafe impl NoComment for () {}
    //~^ undocumented_unsafe_blocks

    unsafe trait InlineComment {}

    /* SAFETY: */ unsafe impl InlineComment for () {}
    //~^ undocumented_unsafe_blocks

    unsafe trait TrailingComment {}

    unsafe impl TrailingComment for () {} // SAFETY:
    //~^ undocumented_unsafe_blocks

    unsafe trait Interference {}
    // SAFETY:
    const BIG_NUMBER: i32 = 1000000;
    //~^ unnecessary_safety_comment
    unsafe impl Interference for () {}
    //~^ undocumented_unsafe_blocks
}

unsafe trait ImplInFn {}

fn impl_in_fn() {
    // error
    unsafe impl ImplInFn for () {}
    //~^ undocumented_unsafe_blocks

    // SAFETY: ok
    unsafe impl ImplInFn for (i32) {}
}

unsafe trait CrateRoot {}

// error
unsafe impl CrateRoot for () {}
//~^ undocumented_unsafe_blocks

// SAFETY: ok
unsafe impl CrateRoot for (i32) {}

fn nested_block_separation_issue_9142() {
    // SAFETY: ok
    let _ =
        // we need this comment to avoid rustfmt putting
        // it all on one line
        unsafe {};
    //~[disabled]^ undocumented_unsafe_blocks

    // SAFETY: this is more than one level away, so it should warn
    let _ = {
        //~^ unnecessary_safety_comment
        if unsafe { true } {
            //~^ undocumented_unsafe_blocks
            todo!();
        } else {
            let bar = unsafe {};
            //~^ undocumented_unsafe_blocks
            todo!();
            bar
        }
    };
}

pub unsafe fn a_function_with_a_very_long_name_to_break_the_line() -> u32 {
    1
}

pub const unsafe fn a_const_function_with_a_very_long_name_to_break_the_line() -> u32 {
    2
}

fn separate_line_from_let_issue_10832() {
    // SAFETY: fail ONLY if `accept-comment-above-statement = false`
    let _some_variable_with_a_very_long_name_to_break_the_line =
        unsafe { a_function_with_a_very_long_name_to_break_the_line() };
    //~[disabled]^ undocumented_unsafe_blocks

    // SAFETY: fail ONLY if `accept-comment-above-statement = false`
    const _SOME_CONST_WITH_A_VERY_LONG_NAME_TO_BREAK_THE_LINE: u32 =
        unsafe { a_const_function_with_a_very_long_name_to_break_the_line() };
    //~[disabled]^ undocumented_unsafe_blocks

    // SAFETY: fail ONLY if `accept-comment-above-statement = false`
    static _SOME_STATIC_WITH_A_VERY_LONG_NAME_TO_BREAK_THE_LINE: u32 =
        unsafe { a_const_function_with_a_very_long_name_to_break_the_line() };
    //~[disabled]^ undocumented_unsafe_blocks
}

fn above_expr_attribute_issue_8679<T: Copy>() {
    // SAFETY: fail ONLY if `accept-comment-above-attribute = false`
    #[allow(unsafe_code)]
    unsafe {}
    //~[disabled]^ undocumented_unsafe_blocks

    // SAFETY: fail ONLY if `accept-comment-above-attribute = false`
    #[expect(unsafe_code, reason = "totally safe")]
    unsafe {
        //~[disabled]^ undocumented_unsafe_blocks
        *std::ptr::null::<T>()
    };

    // SAFETY: fail ONLY if `accept-comment-above-attribute = false`
    #[allow(unsafe_code)]
    let _some_variable_with_a_very_long_name_to_break_the_line =
        unsafe { a_function_with_a_very_long_name_to_break_the_line() };
    //~[disabled]^ undocumented_unsafe_blocks

    // SAFETY: fail ONLY if `accept-comment-above-attribute = false`
    #[allow(unsafe_code)]
    const _SOME_CONST_WITH_A_VERY_LONG_NAME_TO_BREAK_THE_LINE: u32 =
        unsafe { a_const_function_with_a_very_long_name_to_break_the_line() };
    //~[disabled]^ undocumented_unsafe_blocks

    // SAFETY: fail ONLY if `accept-comment-above-attribute = false`
    #[allow(unsafe_code)]
    #[allow(non_upper_case_globals)]
    static _some_static_with_a_very_long_name_to_break_the_line: u32 =
        unsafe { a_const_function_with_a_very_long_name_to_break_the_line() };
    //~[disabled]^ undocumented_unsafe_blocks

    // SAFETY:
    #[allow(unsafe_code)]
    // This shouldn't work either
    unsafe {}
    //~[disabled]^ undocumented_unsafe_blocks
}

mod issue_11246 {
    // Safety: foo
    const _: () = unsafe {};

    // Safety: A safety comment
    const FOO: () = unsafe {};

    // Safety: bar
    static BAR: u8 = unsafe { 0 };
}

// Safety: Another safety comment
const FOO: () = unsafe {};

// trait items and impl items
mod issue_11709 {
    trait MyTrait {
        const NO_SAFETY_IN_TRAIT_BUT_IN_IMPL: u8 = unsafe { 0 };
        //~^ ERROR: unsafe block missing a safety comment

        // SAFETY: safe
        const HAS_SAFETY_IN_TRAIT: i32 = unsafe { 1 };

        // SAFETY: unrelated
        unsafe fn unsafe_fn() {}

        const NO_SAFETY_IN_TRAIT: i32 = unsafe { 1 };
        //~^ ERROR: unsafe block missing a safety comment
    }

    struct UnsafeStruct;

    impl MyTrait for UnsafeStruct {
        // SAFETY: safe in this impl
        const NO_SAFETY_IN_TRAIT_BUT_IN_IMPL: u8 = unsafe { 2 };

        const HAS_SAFETY_IN_TRAIT: i32 = unsafe { 3 };
        //~^ ERROR: unsafe block missing a safety comment
    }

    impl UnsafeStruct {
        const NO_SAFETY_IN_IMPL: i32 = unsafe { 1 };
        //~^ ERROR: unsafe block missing a safety comment
    }
}

fn issue_13024() {
    let mut just_a_simple_variable_with_a_very_long_name_that_has_seventy_eight_characters = 0;
    let here_is_another_variable_with_long_name = 100;

    // SAFETY: fail ONLY if `accept-comment-above-statement = false`
    just_a_simple_variable_with_a_very_long_name_that_has_seventy_eight_characters =
        unsafe { here_is_another_variable_with_long_name };
    //~[disabled]^ undocumented_unsafe_blocks
}

// https://docs.rs/time/0.3.36/src/time/offset_date_time.rs.html#35
mod issue_11709_regression {
    use std::num::NonZeroI32;

    struct Date {
        value: NonZeroI32,
    }

    impl Date {
        const unsafe fn __from_ordinal_date_unchecked(year: i32, ordinal: u16) -> Self {
            Self {
                // Safety: The caller must guarantee that `ordinal` is not zero.
                value: unsafe { NonZeroI32::new_unchecked((year << 9) | ordinal as i32) },
            }
        }

        const fn into_julian_day_just_make_this_line_longer(self) -> i32 {
            42
        }
    }

    /// The Julian day of the Unix epoch.
    // SAFETY: fail ONLY if `accept-comment-above-attribute = false`
    #[allow(unsafe_code)]
    const UNIX_EPOCH_JULIAN_DAY: i32 =
        unsafe { Date::__from_ordinal_date_unchecked(1970, 1) }.into_julian_day_just_make_this_line_longer();
    //~[disabled]^ undocumented_unsafe_blocks
}

fn issue_13039() {
    unsafe fn foo() -> usize {
        1234
    }

    fn bar() -> usize {
        1234
    }

    // SAFETY: unnecessary_safety_comment should not trigger here
    _ = unsafe { foo() };

    // SAFETY: unnecessary_safety_comment triggers here
    _ = bar();
    //~^ unnecessary_safety_comment

    // SAFETY: unnecessary_safety_comment should not trigger here
    _ = unsafe { foo() }
}

fn rfl_issue15034() -> i32 {
    unsafe fn h() -> i32 {
        1i32
    }
    // This shouldn't lint with accept-comment-above-attributes! Thus fixing a false positive!
    // SAFETY: My safety comment!
    #[allow(clippy::unnecessary_cast)]
    return unsafe { h() };
    //~[disabled]^ ERROR: unsafe block missing a safety comment
}

fn main() {}
