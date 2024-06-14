//@ check-pass

// Makes sure that `'r#static` is `'static`

const FOO: &'r#static str = "hello, world";

fn main() {}
