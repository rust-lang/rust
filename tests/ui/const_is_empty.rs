#![warn(clippy::const_is_empty)]

fn test_literal() {
    if "".is_empty() {
        //~^ERROR: this expression always evaluates to true
    }
    if "foobar".is_empty() {
        //~^ERROR: this expression always evaluates to false
    }
}

fn test_byte_literal() {
    if b"".is_empty() {
        //~^ERROR: this expression always evaluates to true
    }
    if b"foobar".is_empty() {
        //~^ERROR: this expression always evaluates to false
    }
}

fn test_no_mut() {
    let mut empty = "";
    if empty.is_empty() {
        // No lint because it is mutable
    }
}

fn test_propagated() {
    let empty = "";
    let non_empty = "foobar";
    let empty2 = empty;
    let non_empty2 = non_empty;
    if empty2.is_empty() {
        //~^ERROR: this expression always evaluates to true
    }
    if non_empty2.is_empty() {
        //~^ERROR: this expression always evaluates to false
    }
}

fn main() {
    let value = "foobar";
    let _ = value.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let x = value;
    let _ = x.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = "".is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = b"".is_empty();
    //~^ ERROR: this expression always evaluates to true
}
