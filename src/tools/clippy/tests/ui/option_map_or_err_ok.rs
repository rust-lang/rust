#![warn(clippy::option_map_or_err_ok)]

fn main() {
    let x = Some("a");
    let _ = x.map_or(Err("a"), Ok);
    //~^ ERROR: called `map_or(Err(_), Ok)` on an `Option` value
}
