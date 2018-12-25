fn main() {
    while true {
        true //~  ERROR mismatched types
             //~| expected type `()`
             //~| found type `bool`
             //~| expected (), found bool
    }
}
