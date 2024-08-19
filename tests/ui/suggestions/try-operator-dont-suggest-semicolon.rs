// Regression test for #87051, where a double semicolon was erroneously
// suggested after a `?` operator.

fn main() -> Result<(), ()> {
    a(|| {
        b()
        //~^ ERROR: mismatched types [E0308]
        //~| NOTE: expected `()`, found `i32`
        //~| HELP: consider using a semicolon here
    })?;

    // Here, we do want to suggest a semicolon:
    let x = Ok(42);
    if true {
    //~^ NOTE: expected this to be `()`
        x?
        //~^ ERROR: mismatched types [E0308]
        //~| NOTE: expected `()`, found integer
        //~| HELP: consider using a semicolon here
    }
    //~^ HELP: consider using a semicolon here

    Ok(())
}

fn a<F>(f: F) -> Result<(), ()> where F: FnMut() { Ok(()) }
fn b() -> i32 { 42 }
