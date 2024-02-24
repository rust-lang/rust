//@ check-pass

// This tests the bug in #70814, where the unused_braces lint triggered on the following code
// without providing a span.

#![deny(unused_braces)]

fn main() {
    {
        {
            use std;
        }
    }
}
