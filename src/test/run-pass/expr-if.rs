


// -*- rust -*-

// Tests for if as expressions
fn test_if() {
    let bool rs = if (true) { true } else { false };
    assert (rs);
}

fn test_else() {
    let bool rs = if (false) { false } else { true };
    assert (rs);
}

fn test_elseif1() {
    let bool rs = if (true) { true } else if (true) { false } else { false };
    assert (rs);
}

fn test_elseif2() {
    let bool rs =
        if (false) { false } else if (true) { true } else { false };
    assert (rs);
}

fn test_elseif3() {
    let bool rs =
        if (false) { false } else if (false) { false } else { true };
    assert (rs);
}

fn test_inferrence() {
    auto rs = if (true) { true } else { false };
    assert (rs);
}

fn test_if_as_if_condition() {
    auto rs1 =
        if (if (false) { false } else { true }) { true } else { false };
    assert (rs1);
    auto rs2 =
        if (if (true) { false } else { true }) { false } else { true };
    assert (rs2);
}

fn test_if_as_block_result() {
    auto rs =
        if (true) { if (false) { false } else { true } } else { false };
    assert (rs);
}

fn main() {
    test_if();
    test_else();
    test_elseif1();
    test_elseif2();
    test_elseif3();
    test_inferrence();
    test_if_as_if_condition();
    test_if_as_block_result();
}