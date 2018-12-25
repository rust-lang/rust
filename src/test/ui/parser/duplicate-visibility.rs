// compile-flags: -Z parse-only

// error-pattern:expected one of `(`, `fn`, `static`, or `type`
extern {
    pub pub fn foo();
}
