fn main() {
    &panic!()
    //~^ ERROR mismatched types
    //~| expected unit type `()`
    //~| found reference `&!`
    //~| expected `()`, found `&!`
}
