fn main() {
    enum Foo {
        Drop = assert_eq!(1, 1),
        //~^ ERROR `if` may be missing an `else` clause
    }
}
