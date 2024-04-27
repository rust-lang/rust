// Exact error being tested isn't relevant, it just needs to be known that it uses Fluent-backed
// diagnostics.

struct Foo {
    val: (),
}

fn foo() -> Foo {
    val: (),
}

fn main() {
    let x = foo();
    x.val == 42;
    let x = {
        val: (),
    };
}
