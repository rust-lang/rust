fn main() {
    enum Foo {
        Drop = assert_eq!(1, 1),
        //~^ ERROR mismatched types [E0308]
    }
}
