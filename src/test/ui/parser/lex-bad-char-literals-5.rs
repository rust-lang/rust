//
// This test needs to the last one appearing in this file as it kills the parser
static c: char =
    '\x10\x10'  //~ ERROR: character literal may only contain one codepoint
                //~| ERROR: mismatched types
;

fn main() {}
