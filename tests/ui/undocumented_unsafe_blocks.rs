// aux-build:proc_macro_unsafe.rs

#![warn(clippy::undocumented_unsafe_blocks, clippy::unnecessary_safety_comment)]
#![allow(clippy::let_unit_value, clippy::missing_safety_doc)]

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
}

fn no_comment() {
    unsafe {}
}

fn no_comment_array() {
    let _ = [unsafe { 14 }, unsafe { 15 }, 42, unsafe { 16 }];
}

fn no_comment_tuple() {
    let _ = (42, unsafe {}, "test", unsafe {});
}

fn no_comment_unary() {
    let _ = *unsafe { &42 };
}

#[allow(clippy::match_single_binding)]
fn no_comment_match() {
    let _ = match unsafe {} {
        _ => {},
    };
}

fn no_comment_addr_of() {
    let _ = &unsafe {};
}

fn no_comment_repeat() {
    let _ = [unsafe {}; 5];
}

fn local_no_comment() {
    let _ = unsafe {};
}

fn no_comment_macro_call() {
    macro_rules! t {
        ($b:expr) => {
            $b
        };
    }

    t!(unsafe {});
}

fn no_comment_macro_def() {
    macro_rules! t {
        () => {
            unsafe {}
        };
    }

    t!();
}

fn trailing_comment() {
    unsafe {} // SAFETY:
}

fn internal_comment() {
    unsafe {
        // SAFETY:
    }
}

fn interference() {
    // SAFETY

    let _ = 42;

    unsafe {};
}

pub fn print_binary_tree() {
    println!("{}", unsafe { String::from_utf8_unchecked(vec![]) });
}

mod unsafe_impl_smoke_test {
    unsafe trait A {}

    // error: no safety comment
    unsafe impl A for () {}

    // Safety: ok
    unsafe impl A for (i32) {}

    mod sub_mod {
        // error:
        unsafe impl B for (u32) {}
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
        };
    }

    // ok
    no_safety_comment!(());

    // error
    unsafe impl T for (i32) {}

    // ok
    no_safety_comment!(u32);

    // error
    unsafe impl T for (bool) {}
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

    unsafe trait InlineComment {}

    /* SAFETY: */ unsafe impl InlineComment for () {}

    unsafe trait TrailingComment {}

    unsafe impl TrailingComment for () {} // SAFETY:

    unsafe trait Interference {}
    // SAFETY:
    const BIG_NUMBER: i32 = 1000000;
    unsafe impl Interference for () {}
}

mod unsafe_items_invalid_comment {
    // SAFETY:
    const CONST: u32 = 0;
    // SAFETY:
    static STATIC: u32 = 0;
    // SAFETY:
    struct Struct;
    // SAFETY:
    enum Enum {}
    // SAFETY:
    mod module {}
}

unsafe trait ImplInFn {}

fn impl_in_fn() {
    // error
    unsafe impl ImplInFn for () {}

    // SAFETY: ok
    unsafe impl ImplInFn for (i32) {}
}

unsafe trait CrateRoot {}

// error
unsafe impl CrateRoot for () {}

// SAFETY: ok
unsafe impl CrateRoot for (i32) {}

fn issue_9142() {
    // SAFETY: ok
    let _ =
        // we need this comment to avoid rustfmt putting
        // it all on one line
        unsafe {};

    // SAFETY: this is more than one level away, so it should warn
    let _ = {
        if unsafe { true } {
            todo!();
        } else {
            let bar = unsafe {};
            todo!();
            bar
        }
    };
}

mod unnecessary_from_macro {
    trait T {}

    macro_rules! no_safety_comment {
        ($t:ty) => {
            impl T for $t {}
        };
    }

    // FIXME: This is not caught
    // Safety: unnecessary
    no_safety_comment!(());

    macro_rules! with_safety_comment {
        ($t:ty) => {
            // Safety: unnecessary
            impl T for $t {}
        };
    }

    with_safety_comment!(i32);
}

fn unnecessary_on_stmt_and_expr() -> u32 {
    // SAFETY: unnecessary
    let num = 42;

    // SAFETY: unnecessary
    24
}

fn main() {}
