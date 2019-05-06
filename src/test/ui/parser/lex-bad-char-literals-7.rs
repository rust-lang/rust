// compile-flags: -Z continue-parse-after-error
fn main() {
    let _: char = '';
    //~^ ERROR: empty character literal
    let _: char = '\u{}';
    //~^ ERROR: empty unicode escape (must have at least 1 hex digit)

    // Next two are OK, but may befool error recovery
    let _ = '/';
    let _ = b'/';

    let _ = ' hello // here's a comment
    //~^ ERROR: unterminated character literal
}
