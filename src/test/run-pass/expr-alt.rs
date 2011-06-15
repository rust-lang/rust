


// -*- rust -*-

// Tests for using alt as an expression
fn test_basic() {
    let bool res = alt (true) { case (true) { true } case (false) { false } };
    assert (res);
    res = alt (false) { case (true) { false } case (false) { true } };
    assert (res);
}

fn test_inferrence() {
    auto res = alt (true) { case (true) { true } case (false) { false } };
    assert (res);
}

fn test_alt_as_alt_head() {
    // Yeah, this is kind of confusing ...

    auto res =
        alt (alt (false) { case (true) { true } case (false) { false } }) {
            case (true) { false }
            case (false) { true }
        };
    assert (res);
}

fn test_alt_as_block_result() {
    auto res =
        alt (false) {
            case (true) { false }
            case (false) {
                alt (true) { case (true) { true } case (false) { false } }
            }
        };
    assert (res);
}

fn main() {
    test_basic();
    test_inferrence();
    test_alt_as_alt_head();
    test_alt_as_block_result();
}