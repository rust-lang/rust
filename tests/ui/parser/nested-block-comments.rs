//! Test that nested block comments are properly supported by the parser.
//!
//! See <https://github.com/rust-lang/rust/issues/66>.

//@ run-pass

/* This test checks that nested comments are supported

   /* This is a nested comment
      /* And this is even more deeply nested */
      Back to the first level of nesting
   */

   /* Another nested comment at the same level */
*/

/* Additional test cases for nested comments */

#[rustfmt::skip]
/*
    /* Level 1
        /* Level 2
            /* Level 3 */
         */
    */
*/

pub fn main() {
    // Check that code after nested comments works correctly
    let _x = 42;

    /* Even inline /* nested */ comments work */
    let _y = /* nested /* comment */ test */ 100;
}
