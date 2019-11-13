fn main() {
    &panic!()
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found reference `&_`
    //~| expected (), found reference
}
