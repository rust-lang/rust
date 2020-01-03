// edition:2018

fn main() {}

async fn func<'a>() -> Result<(), &'a str> {
    let s = String::new();

    let b = &s[..];

    Err(b)?; //~ ERROR cannot return value referencing local variable `s`

    Ok(())
}
