#![deny(dependency_on_unit_never_type_fallback)]

fn create_ok_default<C>() -> Result<C, ()>
where
    C: Default,
{
    Ok(C::default())
}

fn main() -> Result<(), ()> {
    //~^ ERROR this function depends on never type fallback being `()`
    //~| WARN this was previously accepted by the compiler but is being phased out
    let (returned_value, _) = (|| {
        let created = create_ok_default()?;
        Ok((created, ()))
    })()?;

    let _ = format_args!("{:?}", returned_value);
    Ok(())
}
