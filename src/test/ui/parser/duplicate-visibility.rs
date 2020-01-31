// error-pattern: expected one of `(`, `async`, `const`, `extern`, `fn`

fn main() {}

extern {
    pub pub fn foo();
}
