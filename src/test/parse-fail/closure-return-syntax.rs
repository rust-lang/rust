// Test that we cannot parse a closure with an explicit return type
// unless it uses braces.

fn main() {
    let x = || -> i32 22;
    //~^ ERROR expected one of `!`, `(`, `+`, `::`, `<`, or `{`, found `22`
}
