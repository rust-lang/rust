fn main() { //~ NOTE expected `()` because of default return type
    &panic!()
    //~^ ERROR mismatched types
    //~| NOTE expected unit type `()`
    //~| NOTE found reference `&_`
    //~| NOTE expected `()`, found `&_`
}
