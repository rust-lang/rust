fn foo(x: @int) -> fn@() -> &static/int {
    fn@() -> &static/int {&*x} //~ ERROR illegal borrow
}

fn bar(x: @int) -> fn@() -> &int {
    fn@() -> &int {&*x} //~ ERROR illegal borrow
}

fn zed(x: @int) -> fn@() -> int {
    fn@() -> int {*&*x}
}

fn main() {
}