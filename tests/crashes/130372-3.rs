//@ known-bug: rust-lang/rust#130372

fn bar() -> impl Fn() {
    wrap()
}

fn wrap(...: impl ...) -> impl Fn() {}
