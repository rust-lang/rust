//xfail-stage0
//xfail-stage1
//xfail-stage2

// Reduced test case for issue #543.
fn main() {

    obj a() {
        fn foo() -> int {
            ret 2;
        }
    }

    auto my_a = a();

    auto my_b = obj() {
        fn foo() -> int {
            ret 3;
        }
        with my_a
    };

    assert (my_b.foo() == 3);
}
