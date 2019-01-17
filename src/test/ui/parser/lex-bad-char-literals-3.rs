// This test needs to the last one appearing in this file as it kills the parser
static c: char =
    '●●' //~ ERROR: character literal may only contain one codepoint
;

fn main() {
    let ch: &str = '●●'; //~ ERROR: character literal may only contain one codepoint
    //~^ ERROR: mismatched types
}
