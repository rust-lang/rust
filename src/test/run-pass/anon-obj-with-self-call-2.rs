// Reduced test case for issue #540.
fn main() {
    obj a() {
        fn foo() -> int { ret 2; }
    }

    let my_a = a();
    let my_b =
        obj () {
            fn baz() -> int { ret self.foo(); }
            with
            my_a
        };

    assert (my_b.baz() == 2);
}
