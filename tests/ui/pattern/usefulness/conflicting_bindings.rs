//@ edition: 2024

#![feature(if_let_guard)]

fn main() {
    let mut x = Some(String::new());
    let ref mut y @ ref mut z = x;
    //~^ ERROR: mutable more than once
    let Some(ref mut y @ ref mut z) = x else { return };
    //~^ ERROR: mutable more than once
    if let Some(ref mut y @ ref mut z) = x {}
    //~^ ERROR: mutable more than once
    if let Some(ref mut y @ ref mut z) = x && true {}
    //~^ ERROR: mutable more than once
    if let Some(_) = Some(()) && let Some(ref mut y @ ref mut z) = x && true {}
    //~^ ERROR: mutable more than once
    while let Some(ref mut y @ ref mut z) = x {}
    //~^ ERROR: mutable more than once
    while let Some(ref mut y @ ref mut z) = x && true {}
    //~^ ERROR: mutable more than once
    match x {
        ref mut y @ ref mut z => {} //~ ERROR: mutable more than once
    }
    match () {
        () if let Some(ref mut y @ ref mut z) = x => {} //~ ERROR: mutable more than once
        _ => {}
    }
}
