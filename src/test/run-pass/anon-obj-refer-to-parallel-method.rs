//xfail-stage1
//xfail-stage2
//xfail-stage3

// Test case for issue #822.
fn main() {
    obj a() {
        fn foo() -> int {
            ret 2;
        }
    }

    let my_a = a();

    let my_b = obj() {
        fn bar() -> int {
            ret self.baz();
        }
        fn baz() -> int {
            ret 3;
        }
        with my_a
    };
}
