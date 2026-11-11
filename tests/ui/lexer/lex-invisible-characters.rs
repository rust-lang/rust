// Provide extra help when a user has an invisible character in their code

fn main​() {
    //~^ ERROR unknown start of token: \u{200b}
    //~| HELP invisible characters like '\u{200b}' are not usually visible in text editors
}
