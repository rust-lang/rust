// Test the parser for the "while parsing this or-pattern..." label here.

fn main() {
    match Some(42) {
        Some(42) | .=. => {} //~ ERROR expected pattern, found `.`
        //~^ NOTE while parsing this or-pattern starting here
        //~| NOTE expected pattern
    }
}
