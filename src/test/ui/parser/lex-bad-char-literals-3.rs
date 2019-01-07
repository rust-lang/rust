// This test needs to the last one appearing in this file as it kills the parser
static c: char =
    '●●' //~ ERROR: character literal may only contain one codepoint
         //~| ERROR: mismatched types
;

fn main() {}
