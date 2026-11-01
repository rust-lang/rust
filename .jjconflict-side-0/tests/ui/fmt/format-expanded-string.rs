//@ proc-macro: format-string-proc-macro.rs

#[macro_use]
extern crate format_string_proc_macro;


// If the format string is another macro invocation, rustc would previously
// compute nonsensical spans, such as:
//
//   error: invalid format string: unmatched `}` found
//    --> test.rs:2:17
//     |
//   2 |     format!(concat!("abc}"));
//     |                 ^ unmatched `}` in format string
//
// This test checks that this behavior has been fixed.

fn main() {
    format!(concat!("abc}"));
    //~^ ERROR: invalid format string: unmatched `}` found

    format!(err_with_input_span!(""));
    //~^ ERROR: invalid format string: unmatched `}` found
}
