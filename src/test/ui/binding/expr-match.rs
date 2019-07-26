// run-pass




// Tests for using match as an expression

fn test_basic() {
    let mut rs: bool = match true { true => { true } false => { false } };
    assert!((rs));
    rs = match false { true => { false } false => { true } };
    assert!((rs));
}

fn test_inferrence() {
    let rs = match true { true => { true } false => { false } };
    assert!((rs));
}

fn test_alt_as_alt_head() {
    // Yeah, this is kind of confusing ...

    let rs =
        match match false { true => { true } false => { false } } {
          true => { false }
          false => { true }
        };
    assert!((rs));
}

fn test_alt_as_block_result() {
    let rs =
        match false {
          true => { false }
          false => { match true { true => { true } false => { false } } }
        };
    assert!((rs));
}

pub fn main() {
    test_basic();
    test_inferrence();
    test_alt_as_alt_head();
    test_alt_as_block_result();
}
