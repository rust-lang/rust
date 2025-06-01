//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

fn ok() -> Result<Option<bool>, ()> {
    Ok(Some(true))
}

fn main() {
    match ok() {
        Ok(x) if let Err(_) = x => {},
        //~^ ERROR mismatched types
        Ok(x) if let 0 = x => {},
        //~^ ERROR mismatched types
        _ => {}
    }
}
