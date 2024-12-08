fn _f(s: &str) -> Option<()> {
    let _ = s[1..].splitn(2, '.').next()?;
    //~^ ERROR: unnecessary use of `splitn`
    //~| NOTE: `-D clippy::needless-splitn` implied by `-D warnings`
    Some(())
}

fn main() {}
