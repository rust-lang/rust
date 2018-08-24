fn main() {
    &panic!()
    //~^ ERROR mismatched types
    //~| expected type `()`
    //~| found type `&_`
    //~| expected (), found reference
}
