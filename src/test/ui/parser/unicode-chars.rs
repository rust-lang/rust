fn main() {
    let y = 0;
    //~^ ERROR unknown start of token: \u{37e}
    //~^^ HELP Unicode character ';' (Greek Question Mark) looks like ';' (Semicolon), but it is not
}
