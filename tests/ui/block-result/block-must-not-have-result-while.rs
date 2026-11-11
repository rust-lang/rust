//@ dont-require-annotations: NOTE

fn main() {
    while true { //~ WARN denote infinite loops with
        true //~  ERROR mismatched types
             //~| NOTE expected `()`, found `bool`
    }
}
