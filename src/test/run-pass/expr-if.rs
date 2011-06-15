


// -*- rust -*-

// Tests for if as expressions
fn test_if() {
    let bool res = if (true) { true } else { false };
    assert (res);
}

fn test_else() {
    let bool res = if (false) { false } else { true };
    assert (res);
}

fn test_elseif1() {
    let bool res = if (true) { true } else if (true) { false } else { false };
    assert (res);
}

fn test_elseif2() {
    let bool res =
        if (false) { false } else if (true) { true } else { false };
    assert (res);
}

fn test_elseif3() {
    let bool res =
        if (false) { false } else if (false) { false } else { true };
    assert (res);
}

fn test_inferrence() {
    auto res = if (true) { true } else { false };
    assert (res);
}

fn test_if_as_if_condition() {
    auto res1 =
        if (if (false) { false } else { true }) { true } else { false };
    assert (res1);
    auto res2 =
        if (if (true) { false } else { true }) { false } else { true };
    assert (res2);
}

fn test_if_as_block_result() {
    auto res =
        if (true) { if (false) { false } else { true } } else { false };
    assert (res);
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