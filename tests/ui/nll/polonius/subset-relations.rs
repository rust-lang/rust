// Checks that Polonius can compute cases of universal regions errors:
// "illegal subset relation errors", cases where analysis finds that
// two free regions outlive each other, without any evidence that this
// relation holds.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: polonius legacy
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] compile-flags: -Z polonius=legacy

// returning `y` requires that `'b: 'a`, but it's not known to be true
fn missing_subset<'a, 'b>(x: &'a u32, y: &'b u32) -> &'a u32 {
    y //~ ERROR
}

// `'b: 'a` is explicitly declared
fn valid_subset<'a, 'b: 'a>(x: &'a u32, y: &'b u32) -> &'a u32 {
    y
}

// because of `x`, it is implied that `'b: 'a` holds
fn implied_bounds_subset<'a, 'b>(x: &'a &'b mut u32) -> &'a u32 {
    x
}

// `'b: 'a` is declared, and `'a: 'c` is known via implied bounds:
// `'b: 'c` is therefore known to hold transitively
fn transitively_valid_subset<'a, 'b: 'a, 'c>(x: &'c &'a u32, y: &'b u32) -> &'c u32 {
    y
}

fn main() {}
