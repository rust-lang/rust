const fn foo() -> Result<(), ()> {
    Err(())?;
    Ok(())
}
