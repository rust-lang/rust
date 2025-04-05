fn main() {
    while true { //~ WARN denote infinite loops with
        true //~  ERROR mismatched types
             //~| NOTE_NONVIRAL expected `()`, found `bool`
    }
}
