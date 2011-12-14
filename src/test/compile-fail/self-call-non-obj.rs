// error-pattern:unresolved name: self

// Fix for issue #707.
fn main() {

    fn foo() -> int { ret 3(); }

    self.foo();

}
