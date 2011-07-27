


// -*- rust -*-

// Tests for using alt as an expression
fn test_basic() {
    let rs: bool = alt true { true { true } false { false } };
    assert (rs);
    rs = alt false { true { false } false { true } };
    assert (rs);
}

fn test_inferrence() {
    let rs = alt true { true { true } false { false } };
    assert (rs);
}

fn test_alt_as_alt_head() {
    // Yeah, this is kind of confusing ...

    let rs =
        alt alt false { true { true } false { false } } {
          true { false }
          false { true }
        };
    assert (rs);
}

fn test_alt_as_block_result() {
    let rs =
        alt false {
          true { false }
          false { alt true { true { true } false { false } } }
        };
    assert (rs);
}

fn main() {
    test_basic();
    test_inferrence();
    test_alt_as_alt_head();
    test_alt_as_block_result();
}