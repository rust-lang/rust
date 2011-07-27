


// xfail-stage0
fn test_if_fail() { let x = if false { fail } else { 10 }; assert (x == 10); }

fn test_else_fail() {
    let x = if true { 10 } else { fail };
    assert (x == 10);
}

fn test_elseif_fail() {
    let x = if false { 0 } else if (false) { fail } else { 10 };
    assert (x == 10);
}

fn main() { test_if_fail(); test_else_fail(); test_elseif_fail(); }