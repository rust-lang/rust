fn main() {
    enum Foo {
        Drop = assert_eq!(1, 1)
        //~^ ERROR if may be missing an else clause
        //~| ERROR `match` is not allowed in a `const`
        //~| ERROR `match` is not allowed in a `const`
        //~| ERROR `if` is not allowed in a `const`
    }
}
