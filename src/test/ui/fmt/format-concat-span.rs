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
}
