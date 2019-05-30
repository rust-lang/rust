fn foo() {}

fn main() {
    let f: &dyn Fn() = &foo;
    f();
}
