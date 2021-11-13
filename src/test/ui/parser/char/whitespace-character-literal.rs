// This tests that the error generated when a character literal has multiple
// characters in it contains a note about non-printing characters.

fn main() {
    // <hair space>x<zero width space>
    let _hair_space_around = ' x​';
    //~^ ERROR: character literal may only contain one codepoint
    //~| NOTE: there are non-printing characters, the full sequence is `\u{200a}x\u{200b}`
}
