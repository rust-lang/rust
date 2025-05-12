//@ edition:2018
//
// Regression test for issue #67765
// Tests that we point at the proper location when giving
// a lifetime error.
fn main() {}

async fn func<'a>() -> Result<(), &'a str> {
    let s = String::new();

    let b = &s[..];

    Err(b)?; //~ ERROR cannot return value referencing local variable `s`

    Ok(())
}
