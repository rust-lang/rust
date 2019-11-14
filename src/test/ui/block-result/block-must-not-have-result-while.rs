fn main() {
    while true { //~ WARN denote infinite loops with
        true //~  ERROR mismatched types
             //~| expected unit type `()`
             //~| found type `bool`
             //~| expected (), found bool
    }
}
