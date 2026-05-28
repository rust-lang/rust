//@ check-pass

// Check that the string literal in `extern lit` will accept raw strings.

fn main() {}

extern r#"C"# fn foo() {}

extern r#"C"# {
    fn bar();
}

type T = extern r#"C"# fn();
