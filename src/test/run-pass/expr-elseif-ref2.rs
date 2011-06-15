


// xfail-stage0

// Regression test for issue #388
fn main() {
    auto x = if (false) { [0u] } else if (true) { [10u] } else { [0u] };
}