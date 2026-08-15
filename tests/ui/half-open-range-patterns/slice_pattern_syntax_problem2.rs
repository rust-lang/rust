//@ run-pass

fn main() {
    let xs = [13, 1, 5, 2, 3, 1, 21, 8];
    if let [3..=14, ..] = xs {
        /* this variant must pass for now, unfortunately.
         * This test is included here to help inform a future plan for these.
         */
    };
}
