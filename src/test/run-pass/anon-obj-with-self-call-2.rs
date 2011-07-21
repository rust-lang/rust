// Reduced test case for issue #540.
fn main() {
    obj a() {
        fn foo() -> int {
            ret 2;
        }
    }

    auto my_a = a();
    auto my_b = obj {
        fn baz() -> int {
            ret self.foo();
        }
        with my_a
    };

    assert (my_b.baz() == 2);
}
