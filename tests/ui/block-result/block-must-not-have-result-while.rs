fn main() {
    while true { //~ WARN denote infinite loops with
        true //~  ERROR mismatched types
             //~| expected `()`, found `bool`
    }
}
