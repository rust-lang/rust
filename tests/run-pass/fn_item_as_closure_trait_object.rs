fn foo() {}

fn main() {
    let f: &Fn() = &foo;
    f();
}
