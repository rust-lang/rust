//@ check-pass

#![forbid(unreachable_code)]

fn result() -> Result<(), ()> {
    Ok(())?;
    Err(())
}

fn option() -> Option<()> {
    Some(())?;
    None
}

fn main() {
    let _ = result();
    let _ = option();
}
