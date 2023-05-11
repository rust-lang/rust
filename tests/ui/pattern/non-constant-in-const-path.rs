// Checks if we emit `PatternError`s correctly.
// This is also a regression test for #27895 and #68394.

static FOO: u8 = 10;

fn main() {
    let x = 0;
    let 0u8..=x = 0;
    //~^ ERROR: runtime values cannot be referenced in patterns
    let 0u8..=FOO = 0;
    //~^ ERROR: statics cannot be referenced in patterns
    match 1 {
        0 ..= x => {}
        //~^ ERROR: runtime values cannot be referenced in patterns
        0 ..= FOO => {}
        //~^ ERROR: statics cannot be referenced in patterns
    };
}
