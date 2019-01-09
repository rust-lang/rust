fn main() {
    let foo =
        match //~ NOTE while parsing this match expression
        Some(4).unwrap_or_else(5)
        //~^ NOTE expected one of `.`, `?`, `{`, or an operator here
        ; //~ NOTE unexpected token
        //~^ ERROR expected one of `.`, `?`, `{`, or an operator, found `;`

    println!("{}", foo)
}
