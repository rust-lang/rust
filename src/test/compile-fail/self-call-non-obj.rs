//xfail-stage0

// error-pattern:self-call in non-object context

// Fix for issue #707.
fn main() {

    fn foo() -> int {
        ret 3();
    }

    self.foo();

}
