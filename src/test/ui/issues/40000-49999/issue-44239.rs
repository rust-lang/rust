fn main() {
    let n = 0;

    struct Foo;
    impl Foo {
        const N: usize = n;
        //~^ ERROR attempt to use a non-constant value
    }
}
