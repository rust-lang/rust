// Parsing of range patterns

fn main() {
    let 10 ..= makropulos!() = 12;
    //~^ ERROR expected one of `::`, `:`, `;`, `<`, `=`, or `|`, found `!`
}
