// xfail-stage1
// xfail-stage2
// xfail-stage3

// Test case for issue #758.
obj foo() { fn f() { } }

fn main() {
    let my_foo = foo();
    let f = my_foo.f;
}
