#![warn(clippy::undocumented_unsafe_blocks)]

// Valid comments

fn nested_local() {
    let _ = {
        let _ = {
            // Safety:
            let _ = unsafe {};
        };
    };
}

fn deep_nest() {
    let _ = {
        let _ = {
            // Safety:
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

                                // Safety:
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

    // Safety:
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
    // Safety:

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
    /* Safety: */

    unsafe {}
}

#[rustfmt::skip]
fn inline_block_comment() {
    /* Safety: */unsafe {}
}

fn block_comment_with_extras() {
    /* This is a description
     * Safety:
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
    // This is a test. Safety:
    unsafe {}
}

fn local_line_comment() {
    // Safety:
    let _ = unsafe {};
}

fn local_block_comment() {
    /* Safety: */
    let _ = unsafe {};
}

fn comment_array() {
    // Safety:
    let _ = [unsafe { 14 }, unsafe { 15 }, 42, unsafe { 16 }];
}

fn comment_tuple() {
    // Safety:
    let _ = (42, unsafe {}, "test", unsafe {});
}

fn comment_unary() {
    // Safety:
    let _ = *unsafe { &42 };
}

#[allow(clippy::match_single_binding)]
fn comment_match() {
    // Safety:
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
        // Safety:
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
    // ॐ᧻໒ Safety: ௵∰
    unsafe {};
}

fn local_commented_block() {
    let _ =
        // Safety:
        unsafe {};
}

fn local_nest() {
    // Safety:
    let _ = [(42, unsafe {}, unsafe {}), (52, unsafe {}, unsafe {})];
}

// Invalid comments

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
    unsafe {} // Safety:
}

fn internal_comment() {
    unsafe {
        // Safety:
    }
}

fn interference() {
    // Safety

    let _ = 42;

    unsafe {};
}

fn main() {}
