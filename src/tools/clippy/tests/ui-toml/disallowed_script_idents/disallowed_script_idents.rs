#![warn(clippy::disallowed_script_idents)]
fn main() {
    let счётчик = 10;
    let カウンタ = 10;
    //~^ ERROR: identifier `カウンタ` has a Unicode script that is not allowed by configuration
}
