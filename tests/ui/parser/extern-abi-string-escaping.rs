//@ check-pass

// Check that the string literal in `extern lit` will escapes.

fn main() {}

extern "\x43" fn foo() {}

extern "\x43" {
    fn bar();
}

type T = extern "\x43" fn();
