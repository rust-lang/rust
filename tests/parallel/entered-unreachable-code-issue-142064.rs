// Test for #142064, which causes a ice bug, "internal error: entered unreachable code".

//@ compile-flags: -Zthreads=2
//@ edition: 2024

#![crate_type = "lib"]
trait A {
    fn foo() -> A;
}
trait B {
    fn foo() -> A;
}
