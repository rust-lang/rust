fn main() {
    let _: char = '';
    //~^ ERROR: empty character literal
    let _: char = '\u{}';
    //~^ ERROR: empty unicode escape

    // Next two are OK, but may befool error recovery
    let _ = '/';
    let _ = b'/';

    let _ = ' hello // here's a comment
    //~^ ERROR: unterminated character literal
}
