#![warn(clippy::manual_unwrap_or_default)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn main() {
    let x: Option<Vec<String>> = None;
    match x {
        //~^ ERROR: match can be simplified with `.unwrap_or_default()`
        Some(v) => v,
        None => Vec::default(),
    };

    let x: Option<Vec<String>> = None;
    match x {
        //~^ ERROR: match can be simplified with `.unwrap_or_default()`
        Some(v) => v,
        _ => Vec::default(),
    };

    let x: Option<String> = None;
    match x {
        //~^ ERROR: match can be simplified with `.unwrap_or_default()`
        Some(v) => v,
        None => String::new(),
    };

    let x: Option<Vec<String>> = None;
    match x {
        //~^ ERROR: match can be simplified with `.unwrap_or_default()`
        None => Vec::default(),
        Some(v) => v,
    };

    let x: Option<Vec<String>> = None;
    if let Some(v) = x {
        //~^ ERROR: if let can be simplified with `.unwrap_or_default()`
        v
    } else {
        Vec::default()
    };
}
