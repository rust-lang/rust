// Regression test for #115930.
// All of these closures are identical, and should produce identical output in
// the coverage report. However, an unstable sort was causing them to be treated
// inconsistently when preparing coverage spans.

#[rustfmt::skip]
fn main() {
    let truthy = std::env::args().len() == 1;

    let a
        =
        |
        |
        if truthy { true } else { false };

    a();
    if truthy { a(); }

    let b
        =
        |
        |
        if truthy { true } else { false };

    b();
    if truthy { b(); }

    let c
        =
        |
        |
        if truthy { true } else { false };

    c();
    if truthy { c(); }

    let d
        =
        |
        |
        if truthy { true } else { false };

    d();
    if truthy { d(); }
}
