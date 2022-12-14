fn main() {
    let x: &str = 'ab';
    //~^ ERROR: character literal may only contain one codepoint
    let y: char = 'cd';
    //~^ ERROR: character literal may only contain one codepoint
    let z = 'ef';
    //~^ ERROR: character literal may only contain one codepoint

    if x == y {}
    //~^ ERROR: can't compare `&str` with `char`
    if y == z {}  // no error here
    if x == z {}
    //~^ ERROR: can't compare `&str` with `char`

    let a: usize = "";
    //~^ ERROR: mismatched types
}
