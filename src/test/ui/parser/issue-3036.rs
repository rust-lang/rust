// Testing that semicolon tokens are printed correctly in errors

fn main()
{
    let x = 3
} //~ ERROR: expected one of `.`, `;`, `?`, or an operator, found `}`
