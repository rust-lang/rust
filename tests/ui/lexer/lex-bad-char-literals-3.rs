static c: char = '●●';
//~^ ERROR: character literal may only contain one codepoint

fn main() {
    let ch: &str = '●●';
    //~^ ERROR: character literal may only contain one codepoint
}
