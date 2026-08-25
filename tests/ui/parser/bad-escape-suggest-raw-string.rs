fn main() {
    let ok = r"ab\[c";
    let bad = "ab\[c";
    //~^ ERROR unknown character escape: `[`
    //~| HELP for more information, visit <https://doc.rust-lang.org/reference/tokens.html#literals>
    //~| HELP if you meant to write a literal backslash (perhaps escaping in a regular expression), consider a raw string literal
}
