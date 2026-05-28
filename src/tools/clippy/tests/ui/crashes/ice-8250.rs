fn _f(s: &str) -> Option<()> {
    let _ = s[1..].splitn(2, '.').next()?;
    //~^ needless_splitn

    Some(())
}

fn main() {}
