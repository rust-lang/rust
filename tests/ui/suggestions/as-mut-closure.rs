// Borrowing a container for a closure argument must preserve the expected mutability.
//@ run-rustfix

#![allow(unused_mut)]

fn append(value: &mut String) {
    value.push('!');
}

fn increment(value: &mut i32) {
    *value += 1;
}

fn main() {
    let mut option = Some(String::new());
    let _ = option.map(|arg| append(arg));
    //~^ ERROR mismatched types
    let _ = (&mut option).and_then(|arg| Some(append(arg)));
    //~^ ERROR mismatched types

    let mut result: Result<_, ()> = Ok(String::new());
    let _ = result.map(|arg| append(arg));
    //~^ ERROR mismatched types
    let _ = (&mut result).and_then(|arg| Ok(append(arg)));
    //~^ ERROR mismatched types

    // A shared reference cannot supply `as_mut()`. Borrow the copied argument instead.
    let shared = &Some(0);
    let _ = shared.map(|mut arg| increment(arg));
    //~^ ERROR mismatched types
    let nested = &mut &Some(0);
    let _ = nested.map(|mut arg| increment(arg));
    //~^ ERROR mismatched types
}
