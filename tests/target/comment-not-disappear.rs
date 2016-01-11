// All the comments here should not disappear.

fn a() {
    match x {
        X |
        // A comment
        Y => {}
    };
}

fn b() {
    match x {
        X =>
            // A comment
            y
    }
}

fn c() {
    a() /* ... */;
}

fn foo() -> Vec<i32> {
    (0..11)
        .map(|x|
        // This comment disappears.
        if x % 2 == 0 { x } else { x * 2 })
        .collect()
}

fn d() {
    if true /* and ... */ {
        a();
    }
}

fn calc_page_len(prefix_len: usize, sofar: usize) -> usize {
    2 // page type and flags
    + 1 // stored depth
    + 2 // stored count
    + prefix_len + sofar // sum of size of all the actual items
}
