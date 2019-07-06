fn main() {
    while true { //~ WARN denote infinite loops with
        true //~  ERROR mismatched types
             //~| expected type `()`
             //~| found type `bool`
             //~| expected (), found bool
    }
}
