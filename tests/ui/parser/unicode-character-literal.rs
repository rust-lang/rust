// Regression test for #88684: Improve diagnostics for combining marks
// in character literals.

//@ run-rustfix

fn main() {
    let _spade = '♠️';
    //~^ ERROR: character literal may only contain one codepoint
    //~| NOTE: this `♠` is followed by the combining mark `\u{fe0f}`
    //~| HELP: if you meant to write a string literal, use double quotes

    let _s = 'ṩ̂̊';
    //~^ ERROR: character literal may only contain one codepoint
    //~| NOTE: this `s` is followed by the combining marks `\u{323}\u{307}\u{302}\u{30a}`
    //~| HELP: if you meant to write a string literal, use double quotes

    let _a = 'Å';
    //~^ ERROR: character literal may only contain one codepoint
    //~| NOTE: this `A` is followed by the combining mark `\u{30a}`
    //~| HELP: consider using the normalized form `\u{c5}` of this character
}
