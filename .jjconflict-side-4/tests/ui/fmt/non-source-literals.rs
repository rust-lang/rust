/// Do not point at the format string if it wasn't written in the source.
//@ forbid-output: required by this formatting parameter

#[derive(Debug)]
pub struct NonDisplay;
pub struct NonDebug;

fn main() {
    let _ = format!(concat!("{", "}"), NonDisplay); //~ ERROR
    let _ = format!(concat!("{", "0", "}"), NonDisplay); //~ ERROR
    let _ = format!(concat!("{:", "?}"), NonDebug); //~ ERROR
    let _ = format!(concat!("{", "0", ":?}"), NonDebug); //~ ERROR
}
