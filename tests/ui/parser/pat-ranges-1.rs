// Parsing of range patterns

fn main() {
    let macropus!() ..= 11 = 12; //~ error: expected one of `:`, `;`, `=`, or `|`, found `..=`
}
