#![feature(if_let_guard)]
#![allow(incomplete_features)]

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
