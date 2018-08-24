// compile-flags: -Z parse-only

// error-pattern:expected one of `(`, `fn`, `static`, `type`, or `}` here
extern {
    pub pub fn foo();
}
