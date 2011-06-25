


// -*- rust -*-

// Tests for using alt as an expression
fn test_basic() {
    let bool rs = alt (true) { case (true) { true } case (false) { false } };
    assert (rs);
    rs = alt (false) { case (true) { false } case (false) { true } };
    assert (rs);
}

fn test_inferrence() {
    auto rs = alt (true) { case (true) { true } case (false) { false } };
    assert (rs);
}

fn test_alt_as_alt_head() {
    // Yeah, this is kind of confusing ...

    auto rs =
        alt (alt (false) { case (true) { true } case (false) { false } }) {
            case (true) { false }
            case (false) { true }
        };
    assert (rs);
}

fn test_alt_as_block_result() {
    auto rs =
        alt (false) {
            case (true) { false }
            case (false) {
                alt (true) { case (true) { true } case (false) { false } }
            }
        };
    assert (rs);
}

fn main() {
    test_basic();
    test_inferrence();
    test_alt_as_alt_head();
    test_alt_as_block_result();
}