//@ check-pass
//
// Regression test for issue #86082
//
// Checks that option_env! does not panic on receiving an invalid
// environment variable name.

fn main() {
    option_env!("\0=");
}
