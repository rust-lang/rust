fn main() -> Result<(), ()> {
    vec![Ok(2)].into_iter().collect:<Result<Vec<_>,_>>()?;
    //~^ ERROR expected `::`, found `(`
    Ok(())
}
