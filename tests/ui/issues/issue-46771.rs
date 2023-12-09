fn main() {
    struct Foo;
    (1 .. 2).into_iter().find(|_| Foo(0) == 0); //~ ERROR expected function, found `Foo`
}
